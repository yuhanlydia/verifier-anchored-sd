"""Paper-faithful mapper adapters and nonlinear full-head MLP baseline.

The original Cross-Model KV paper uses all source KV heads from the selected
source layers as the feature vector for every receiver KV head.  Its nonlinear
ablation replaces each independent ridge head with a 3-layer MLP while preserving
the same selected source layers and content-space key treatment.

This module deliberately keeps the runtime mapper independent of the optional
``kvbridge`` package.  Conversion helpers return the pinned kvbridge types when the
extra is installed and small protocol-compatible adapters otherwise, so CPU unit
tests do not need a Git dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .spec_decode.cache_state import CacheState, LayerKV, RotaryFactors


@dataclass(frozen=True)
class _CompatRotaryFactors:
    cos: torch.Tensor
    sin: torch.Tensor
    interleaved: bool = False

    def apply(self, x: torch.Tensor, *, inverse: bool = False) -> torch.Tensor:
        return RotaryFactors(self.cos, self.sin, self.interleaved).apply(x, inverse=inverse)


@dataclass(frozen=True)
class _CompatKVCache:
    keys: tuple[torch.Tensor, ...]
    values: tuple[torch.Tensor, ...]
    rotary: _CompatRotaryFactors | None = None
    keys_are_content: bool = False

    @property
    def shape(self) -> tuple[int, int, int, int, int]:
        batch, heads, tokens, dim = self.keys[0].shape
        return len(self.keys), batch, heads, tokens, dim

    def to(self, device: str | torch.device, *, dtype: torch.dtype | None = None):
        rotary = None
        if self.rotary is not None:
            rotary = _CompatRotaryFactors(
                self.rotary.cos.to(device=device, dtype=dtype),
                self.rotary.sin.to(device=device, dtype=dtype),
                self.rotary.interleaved,
            )
        return replace(
            self,
            keys=tuple(x.to(device=device, dtype=dtype) for x in self.keys),
            values=tuple(x.to(device=device, dtype=dtype) for x in self.values),
            rotary=rotary,
        )

    def to_content_space(self):
        if self.keys_are_content:
            return self
        if self.rotary is None:
            raise ValueError("content-space conversion requires rotary factors")
        return replace(
            self,
            keys=tuple(self.rotary.apply(x, inverse=True) for x in self.keys),
            keys_are_content=True,
        )

    def sample_tokens(self, stride: int):
        if stride <= 0:
            raise ValueError("stride must be positive")
        if stride == 1:
            return self
        rotary = None
        if self.rotary is not None:
            rotary = _CompatRotaryFactors(
                self.rotary.cos[..., ::stride, :],
                self.rotary.sin[..., ::stride, :],
                self.rotary.interleaved,
            )
        return replace(
            self,
            keys=tuple(x[..., ::stride, :] for x in self.keys),
            values=tuple(x[..., ::stride, :] for x in self.values),
            rotary=rotary,
        )


@dataclass(frozen=True)
class _CompatModelSignature:
    model_id: str
    revision: str
    tokenizer_hash: str
    num_layers: int
    num_kv_heads: int
    head_dim: int
    attention_kind: str = "dense"
    architecture: str = "unknown"

    def validate_pair(self, other, *, require_matched_kv: bool = True) -> None:
        if self.tokenizer_hash != other.tokenizer_hash:
            raise ValueError("source and receiver tokenizers differ")
        if require_matched_kv and (
            self.num_kv_heads != other.num_kv_heads or self.head_dim != other.head_dim
        ):
            raise ValueError("matched KV heads/head dimension required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "tokenizer_hash": self.tokenizer_hash,
            "num_layers": self.num_layers,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "attention_kind": self.attention_kind,
            "architecture": self.architecture,
        }


def cache_state_to_kvbridge(cache: CacheState):
    """Convert ``CacheState`` to the pinned kvbridge cache protocol.

    The real kvbridge type is returned when the optional dependency is installed;
    otherwise a protocol-compatible object is used for dependency-free tests.
    """
    rotary = None
    if cache.rotary is not None:
        try:
            from kvbridge.cache import RotaryFactors as ExternalRotary

            rotary = ExternalRotary(
                cache.rotary.cos, cache.rotary.sin, cache.rotary.interleaved
            )
        except ImportError:
            rotary = _CompatRotaryFactors(
                cache.rotary.cos, cache.rotary.sin, cache.rotary.interleaved
            )
    try:
        from kvbridge.cache import KVCache

        return KVCache(
            [layer.key for layer in cache.layers],
            [layer.value for layer in cache.layers],
            rotary=rotary,
            keys_are_content=cache.keys_are_content,
        )
    except ImportError:
        return _CompatKVCache(
            tuple(layer.key for layer in cache.layers),
            tuple(layer.value for layer in cache.layers),
            rotary,
            cache.keys_are_content,
        )


def model_signature_from_manifest(manifest: dict):
    """Build an exact model identity from a modern sequential-capture manifest."""
    model = manifest.get("model")
    if not isinstance(model, dict):
        raise ValueError("capture manifest is missing model metadata")
    required = ("id", "revision", "tokenizer_hash", "layers", "kv_heads", "head_dim")
    missing = [key for key in required if key not in model]
    if missing:
        raise ValueError(f"capture manifest model metadata missing fields: {missing}")
    kwargs = {
        "model_id": str(model["id"]),
        "revision": str(model["revision"]),
        "tokenizer_hash": str(model["tokenizer_hash"]),
        "num_layers": int(model["layers"]),
        "num_kv_heads": int(model["kv_heads"]),
        "head_dim": int(model["head_dim"]),
        "attention_kind": "dense",
        "architecture": str(model.get("architecture", "unknown")),
    }
    try:
        from kvbridge.config import ModelSignature

        return ModelSignature(**kwargs)
    except ImportError:
        return _CompatModelSignature(**kwargs)


@dataclass(frozen=True)
class MLPMapperMetadata:
    """Geometry for the paper's independent full-head MLP receivers."""

    target_layers: int
    draft_layers: int
    target_kv_heads: int
    draft_kv_heads: int
    head_dim: int
    layer_selection: list[list[int]]
    content_space: bool = True
    hidden_dim: int = 1024

    def validate(self) -> None:
        if min(
            self.target_layers,
            self.draft_layers,
            self.target_kv_heads,
            self.draft_kv_heads,
            self.head_dim,
            self.hidden_dim,
        ) <= 0:
            raise ValueError("MLP mapper geometry must be positive")
        if len(self.layer_selection) != self.draft_layers or any(
            not row for row in self.layer_selection
        ):
            raise ValueError("one non-empty source-layer selection is required per draft layer")
        if any(
            index < 0 or index >= self.target_layers
            for row in self.layer_selection
            for index in row
        ):
            raise ValueError("source-layer selection contains an out-of-range layer")

    def feature_width(self, draft_layer: int) -> int:
        return (
            len(self.layer_selection[draft_layer])
            * self.target_kv_heads
            * self.head_dim
        )

    @classmethod
    def from_dict(cls, value: dict) -> "MLPMapperMetadata":
        return cls(**value)


class GroupedHeadMLPMapper:
    """Mathematically independent paper MLPs stored in grouped tensors.

    Every receiver head sees the same full cross-head source feature vector, but
    each ``(receiver layer, K/V, receiver head)`` has its own W1/W2/W3 and biases.
    Grouping only makes inference/training implementation efficient; it does not
    share parameters across heads.
    """

    def __init__(
        self,
        metadata: MLPMapperMetadata,
        w1: torch.Tensor,
        b1: torch.Tensor,
        w2: torch.Tensor,
        b2: torch.Tensor,
        w3: torch.Tensor,
        b3: torch.Tensor,
    ) -> None:
        metadata.validate()
        self.metadata = metadata
        self.w1, self.b1 = w1, b1
        self.w2, self.b2 = w2, b2
        self.w3, self.b3 = w3, b3
        self._validate()

    def _validate(self) -> None:
        m = self.metadata
        prefix = (m.draft_layers, 2, m.draft_kv_heads)
        max_features = max(m.feature_width(layer) for layer in range(m.draft_layers))
        expected = {
            "w1": (*prefix, max_features, m.hidden_dim),
            "b1": (*prefix, m.hidden_dim),
            "w2": (*prefix, m.hidden_dim, m.hidden_dim),
            "b2": (*prefix, m.hidden_dim),
            "w3": (*prefix, m.hidden_dim, m.head_dim),
            "b3": (*prefix, m.head_dim),
        }
        for name, shape in expected.items():
            tensor = getattr(self, name)
            if tuple(tensor.shape) != shape:
                raise ValueError(f"{name} shape {tuple(tensor.shape)} != expected {shape}")
            if not tensor.is_floating_point():
                raise ValueError(f"{name} must be finite floating point")
            # A full MLP can contain billions of weights. Bound temporary
            # isfinite allocations to one receiver head while checking all values.
            for layer in tensor.unbind(0):
                for kind in layer.unbind(0):
                    for head in kind.unbind(0):
                        if not torch.isfinite(head).all():
                            raise ValueError(f"{name} must be finite floating point")

    @property
    def device(self) -> torch.device:
        return self.w1.device

    def to(
        self,
        device: str | torch.device,
        *,
        dtype: torch.dtype | None = None,
    ) -> "GroupedHeadMLPMapper":
        for name in ("w1", "b1", "w2", "b2", "w3", "b3"):
            tensor = getattr(self, name)
            setattr(self, name, tensor.to(device=device, dtype=dtype))
        return self

    @staticmethod
    def _full_features(
        source: CacheState, selected: list[int], kind: int
    ) -> torch.Tensor:
        rows = []
        for layer in selected:
            tensor = source.layers[layer].key if kind == 0 else source.layers[layer].value
            batch, heads, tokens, dim = tensor.shape
            rows.append(tensor.permute(0, 2, 1, 3).reshape(batch, tokens, heads * dim))
        return torch.cat(rows, dim=-1)

    def _map_kind(
        self,
        features: torch.Tensor,
        *,
        layer: int,
        kind: int,
    ) -> torch.Tensor:
        active = features.shape[-1]
        output_dtype = features.dtype
        x = features.float()
        w1 = self.w1[layer, kind, :, :active].to(device=x.device, dtype=torch.float32)
        b1 = self.b1[layer, kind].to(device=x.device, dtype=torch.float32)
        w2 = self.w2[layer, kind].to(device=x.device, dtype=torch.float32)
        b2 = self.b2[layer, kind].to(device=x.device, dtype=torch.float32)
        w3 = self.w3[layer, kind].to(device=x.device, dtype=torch.float32)
        b3 = self.b3[layer, kind].to(device=x.device, dtype=torch.float32)
        hidden = F.relu(torch.einsum("btp,hpf->bhtf", x, w1) + b1[:, None, :])
        hidden = F.relu(
            torch.einsum("bhtf,hfg->bhtg", hidden, w2) + b2[:, None, :]
        )
        output = torch.einsum("bhtf,hfd->bhtd", hidden, w3) + b3[:, None, :]
        return output.to(output_dtype)

    def map(
        self,
        target: CacheState,
        *,
        draft_rotary: RotaryFactors | None = None,
        include_residual: bool = True,
    ) -> CacheState:
        del include_residual  # API compatibility with RidgeKVMapper
        m = self.metadata
        if target.num_layers != m.target_layers:
            raise ValueError("verifier cache layer count does not match MLP mapper")
        if target.kv_heads != m.target_kv_heads or target.head_dim != m.head_dim:
            raise ValueError("verifier KV geometry does not match MLP mapper")
        source = target.to_content_space() if m.content_space else target
        layers: list[LayerKV] = []
        for layer, selected in enumerate(m.layer_selection):
            key_features = self._full_features(source, selected, 0)
            value_features = self._full_features(source, selected, 1)
            layers.append(
                LayerKV(
                    self._map_kind(key_features, layer=layer, kind=0),
                    self._map_kind(value_features, layer=layer, kind=1),
                )
            )
        mapped = CacheState(layers, keys_are_content=m.content_space)
        if m.content_space:
            if draft_rotary is None:
                raise ValueError("content-space MLP mapping requires draft RoPE factors")
            mapped = mapped.apply_rotary(draft_rotary)
        return mapped

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        torch.save(
            {
                "schema_version": 1,
                "metadata": self.metadata.__dict__,
                "w1": self.w1.detach().cpu(),
                "b1": self.b1.detach().cpu(),
                "w2": self.w2.detach().cpu(),
                "b2": self.b2.detach().cpu(),
                "w3": self.w3.detach().cpu(),
                "b3": self.b3.detach().cpu(),
            },
            temporary,
        )
        temporary.replace(destination)

    @classmethod
    def load(
        cls,
        path: str | Path,
        map_location: str | torch.device = "cpu",
    ) -> "GroupedHeadMLPMapper":
        try:
            payload = torch.load(path, map_location=map_location, weights_only=True)
        except Exception as exc:
            raise RuntimeError(f"failed to load MLP mapper {path}") from exc
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise RuntimeError("unsupported MLP mapper schema")
        metadata = MLPMapperMetadata.from_dict(payload["metadata"])
        return cls(
            metadata,
            payload["w1"],
            payload["b1"],
            payload["w2"],
            payload["b2"],
            payload["w3"],
            payload["b3"],
        )
