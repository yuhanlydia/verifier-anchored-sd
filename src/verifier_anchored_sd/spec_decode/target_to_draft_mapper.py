"""Target-to-draft KV mapping plus mergeable low-rank acceptance residuals.

Two affine support patterns are implemented deliberately:

``full``
    The original Cross-Model KV / KVBridge baseline.  Every draft KV head reads all
    verifier KV heads from every selected source layer.

``matched``
    A head-local baseline inspired by CacheBridge.  For matched-KV model pairs,
    draft head ``h`` reads only verifier head ``h`` from the selected source layers.
    This reduces affine input width by the number of KV heads while keeping the
    verifier-anchored speculative-decoding algorithm unchanged.

Keys are mapped in RoPE-free content space and re-rotated with draft-model factors.
Live storage may be BF16; projections accumulate in FP32 for stable ridge numerics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import nn

from .cache_state import CacheState, LayerKV, RotaryFactors

HeadMode = Literal["full", "matched"]


@dataclass
class MapperMetadata:
    target_layers: int
    draft_layers: int
    target_kv_heads: int
    draft_kv_heads: int
    head_dim: int
    layer_selection: list[list[int]]
    lambda_: float
    content_space: bool = True
    head_mode: HeadMode = "full"

    @classmethod
    def from_dict(cls, value: Mapping) -> "MapperMetadata":
        data = dict(value)
        # Backward compatibility is metadata-only. Scientifically invalid old
        # checkpoint tensors still fail the width validation in RidgeKVMapper.
        if "target_kv_heads" not in data and "kv_heads" in data:
            data["target_kv_heads"] = data["kv_heads"]
            data["draft_kv_heads"] = data["kv_heads"]
            data.pop("kv_heads")
        data.setdefault("content_space", False)
        data.setdefault("head_mode", "full")
        return cls(**data)

    def feature_width(self, selected_layers: int) -> int:
        if self.head_mode == "full":
            return selected_layers * self.target_kv_heads * self.head_dim
        if self.head_mode == "matched":
            return selected_layers * self.head_dim
        raise ValueError(f"unsupported head_mode: {self.head_mode}")


class RidgeKVMapper:
    """Affine KV mapper with full-head or matched-head feature support.

    ``weights`` is ``[draft_layers, 2, draft_heads, head_dim, max_input_dim]``.
    Feature order for ``full`` is layer-major, source-head-major, head dimension.
    Feature order for ``matched`` is selected-layer-major within each matching head.
    """

    def __init__(
        self, metadata: MapperMetadata, weights: torch.Tensor, bias: torch.Tensor
    ) -> None:
        if metadata.head_mode not in {"full", "matched"}:
            raise ValueError("head_mode must be 'full' or 'matched'")
        if metadata.head_mode == "matched" and (
            metadata.target_kv_heads != metadata.draft_kv_heads
        ):
            raise ValueError("matched-head mapping requires equal verifier/draft KV-head counts")
        expected_prefix = (
            metadata.draft_layers,
            2,
            metadata.draft_kv_heads,
            metadata.head_dim,
        )
        if tuple(weights.shape[:4]) != expected_prefix:
            raise ValueError(f"unexpected mapper weight shape {weights.shape}")
        if tuple(bias.shape) != expected_prefix:
            raise ValueError(f"unexpected mapper bias shape {bias.shape}")
        required = metadata.feature_width(max(len(x) for x in metadata.layer_selection))
        if weights.shape[-1] < required:
            raise ValueError(
                "mapper weight input width is smaller than the selected feature support"
            )
        self.metadata = metadata
        self.weights = weights
        self.bias = bias
        self.u: torch.Tensor | None = None
        self.v: torch.Tensor | None = None
        self.gate = torch.tensor(0.0, dtype=weights.dtype, device=weights.device)

    @property
    def device(self) -> torch.device:
        return self.weights.device

    @property
    def in_dim(self) -> int:
        return self.weights.shape[-1]

    def to(
        self,
        device: str | torch.device,
        *,
        dtype: torch.dtype | None = None,
    ) -> "RidgeKVMapper":
        self.weights = self.weights.to(device=device, dtype=dtype)
        self.bias = self.bias.to(device=device, dtype=dtype)
        if self.u is not None:
            self.u = nn.Parameter(self.u.to(device=device, dtype=dtype))
        if self.v is not None:
            self.v = nn.Parameter(self.v.to(device=device, dtype=dtype))
        if isinstance(self.gate, nn.Parameter):
            self.gate = nn.Parameter(self.gate.to(device=device, dtype=dtype))
        else:
            self.gate = self.gate.to(device=device, dtype=dtype)
        return self

    def _features(
        self, cache: CacheState, selected: Sequence[int], kind: int
    ) -> torch.Tensor:
        rows = []
        for layer in selected:
            tensor = cache.layers[layer].key if kind == 0 else cache.layers[layer].value
            b, h, t, d = tensor.shape
            if self.metadata.head_mode == "full":
                rows.append(tensor.permute(0, 2, 1, 3).reshape(b, t, h * d))
            else:
                # [B,H,T,D] -> concatenate selected layers on the feature axis,
                # preserving the architecture-indexed head correspondence.
                rows.append(tensor)
        return torch.cat(rows, dim=-1)

    def _map_kind(
        self,
        x: torch.Tensor,
        layer: int,
        kind: int,
        *,
        include_residual: bool,
    ) -> torch.Tensor:
        output_dtype = x.dtype
        active = x.shape[-1]
        compute = x.float()
        w = self.weights[layer, kind, :, :, :active].to(
            device=x.device, dtype=torch.float32
        )
        b = self.bias[layer, kind].to(device=x.device, dtype=torch.float32)
        if self.metadata.head_mode == "full":
            y = torch.einsum("btf,hdf->bhtd", compute, w)
        else:
            y = torch.einsum("bhtf,hdf->bhtd", compute, w)
        y = y + b.view(1, self.metadata.draft_kv_heads, 1, -1)
        if include_residual and self.u is not None and self.v is not None:
            u = self.u[layer, kind].to(device=x.device, dtype=torch.float32)
            v = self.v[layer, kind, :, :, :active].to(
                device=x.device, dtype=torch.float32
            )
            if self.metadata.head_mode == "full":
                residual = torch.einsum("btf,hrf,hdr->bhtd", compute, v, u)
            else:
                residual = torch.einsum("bhtf,hrf,hdr->bhtd", compute, v, u)
            y = y + self.gate.float() * residual
        return y.to(output_dtype)

    def map(
        self,
        target: CacheState,
        *,
        draft_rotary: RotaryFactors | None = None,
        include_residual: bool = True,
    ) -> CacheState:
        if target.num_layers != self.metadata.target_layers:
            raise ValueError("verifier cache layer count does not match mapper")
        if (
            target.kv_heads != self.metadata.target_kv_heads
            or target.head_dim != self.metadata.head_dim
        ):
            raise ValueError("verifier KV geometry does not match mapper")
        source = target.to_content_space() if self.metadata.content_space else target
        result = []
        for dl, selected in enumerate(self.metadata.layer_selection):
            keys = self._features(source, selected, 0)
            values = self._features(source, selected, 1)
            result.append(
                LayerKV(
                    self._map_kind(keys, dl, 0, include_residual=include_residual),
                    self._map_kind(values, dl, 1, include_residual=include_residual),
                )
            )
        mapped = CacheState(result, keys_are_content=self.metadata.content_space)
        if self.metadata.content_space:
            if draft_rotary is None:
                raise ValueError("content-space mapping requires draft-model RoPE factors")
            mapped = mapped.apply_rotary(draft_rotary)
        return mapped

    def add_low_rank(self, rank: int = 8) -> None:
        if rank < 1:
            raise ValueError("rank must be positive")
        shape = (self.metadata.draft_layers, 2, self.metadata.draft_kv_heads)
        self.u = nn.Parameter(
            torch.randn(
                *shape,
                self.metadata.head_dim,
                rank,
                device=self.device,
                dtype=self.weights.dtype,
            )
            * 1e-3
        )
        self.v = nn.Parameter(
            torch.randn(
                *shape,
                rank,
                self.in_dim,
                device=self.device,
                dtype=self.weights.dtype,
            )
            * 1e-3
        )
        self.gate = nn.Parameter(
            torch.tensor(0.0, device=self.device, dtype=self.weights.dtype)
        )

    def merge_residual(self) -> "RidgeKVMapper":
        if self.u is None or self.v is None:
            return self
        delta = torch.einsum("...dr,...rf->...df", self.u, self.v)
        self.weights = self.weights + self.gate * delta
        self.u = self.v = None
        self.gate = torch.tensor(0.0, device=self.device, dtype=self.weights.dtype)
        return self

    def residual_parameters(self):
        if self.u is None or self.v is None:
            raise RuntimeError("call add_low_rank first")
        return [self.u, self.v, self.gate]

    def state_dict(self) -> dict:
        return {
            "weights": self.weights.detach().cpu(),
            "bias": self.bias.detach().cpu(),
            "metadata": self.metadata.__dict__,
            "gate": self.gate.detach().cpu(),
            "u": None if self.u is None else self.u.detach().cpu(),
            "v": None if self.v is None else self.v.detach().cpu(),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping) -> "RidgeKVMapper":
        metadata = MapperMetadata.from_dict(state["metadata"])
        mapper = cls(metadata, state["weights"], state["bias"])
        saved_gate = state.get("gate", mapper.gate)
        saved_u, saved_v = state.get("u"), state.get("v")
        if saved_u is not None and saved_v is not None:
            mapper.u = nn.Parameter(saved_u)
            mapper.v = nn.Parameter(saved_v)
            mapper.gate = nn.Parameter(saved_gate)
        else:
            mapper.gate = saved_gate
        return mapper

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(
        cls, path: str | Path, map_location: str | torch.device = "cpu"
    ) -> "RidgeKVMapper":
        return cls.from_state_dict(
            torch.load(path, map_location=map_location, weights_only=False)
        )

    @classmethod
    def from_kvbridge_artifact(
        cls,
        directory: str | Path,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> "RidgeKVMapper":
        """Import the original full-head KVBridge artifact into runtime format."""
        try:
            from kvbridge.mapper import CrossModelKVMapper
        except ImportError as exc:  # pragma: no cover - optional GPU/E0 environment
            raise RuntimeError(
                "install the kvbridge extra before importing an E0 artifact"
            ) from exc
        external = CrossModelKVMapper.load(directory)
        source = external.source_signature
        draft = external.target_signature
        if source.head_dim != draft.head_dim:
            raise ValueError("verifier and draft head dimensions must match")
        max_in = max(weight.shape[1] for weight in external.key_weights)
        weights = torch.zeros(
            draft.num_layers,
            2,
            draft.num_kv_heads,
            draft.head_dim,
            max_in,
            dtype=dtype,
        )
        bias = torch.zeros(
            draft.num_layers, 2, draft.num_kv_heads, draft.head_dim, dtype=dtype
        )
        for layer in range(draft.num_layers):
            width = external.key_weights[layer].shape[1]
            weights[layer, 0, :, :, :width] = external.key_weights[layer].permute(
                0, 2, 1
            ).to(dtype)
            weights[layer, 1, :, :, :width] = external.value_weights[layer].permute(
                0, 2, 1
            ).to(dtype)
            bias[layer, 0] = external.key_biases[layer].to(dtype)
            bias[layer, 1] = external.value_biases[layer].to(dtype)
        metadata = MapperMetadata(
            target_layers=source.num_layers,
            draft_layers=draft.num_layers,
            target_kv_heads=source.num_kv_heads,
            draft_kv_heads=draft.num_kv_heads,
            head_dim=source.head_dim,
            layer_selection=[list(x) for x in external.selected_layers],
            lambda_=external.config.ridge_alpha,
            content_space=external.config.content_space,
            head_mode="full",
        )
        return cls(metadata, weights, bias)


def fit_ridge_mapper(
    observations: Mapping[tuple[int, int, str], tuple[torch.Tensor, torch.Tensor]],
    *,
    target_layers: int,
    draft_layers: int,
    kv_heads: int,
    head_dim: int,
    layer_selection: Sequence[Sequence[int]],
    lambda_: float = 0.01,
    content_space: bool = False,
    head_mode: HeadMode = "full",
) -> RidgeKVMapper:
    """Small synthetic/test fitter with caller-supplied source-layer selection."""
    if lambda_ < 0:
        raise ValueError("lambda_ must be non-negative")
    if head_mode not in {"full", "matched"}:
        raise ValueError("head_mode must be 'full' or 'matched'")
    selection = [list(x) for x in layer_selection]
    if len(selection) != draft_layers:
        raise ValueError("one source-layer selection is required per draft layer")
    feature_heads = kv_heads if head_mode == "full" else 1
    max_in = max(len(x) for x in selection) * feature_heads * head_dim
    weights = torch.zeros(draft_layers, 2, kv_heads, head_dim, max_in)
    bias = torch.zeros(draft_layers, 2, kv_heads, head_dim)
    for dl in range(draft_layers):
        expected_in = len(selection[dl]) * feature_heads * head_dim
        for h in range(kv_heads):
            for kind, name in enumerate(("k", "v")):
                x, y = observations[(dl, h, name)]
                x, y = x.float(), y.float()
                if (
                    x.ndim != 2
                    or y.ndim != 2
                    or x.shape[0] != y.shape[0]
                    or y.shape[1] != head_dim
                ):
                    raise ValueError(f"invalid observation shape for {(dl, h, name)}")
                if x.shape[1] != expected_in:
                    raise ValueError(
                        f"expected {expected_in} input features, got {x.shape[1]}"
                    )
                if x.shape[0] <= x.shape[1]:
                    raise ValueError(
                        "ridge fit is underdetermined: observations must exceed feature dimension"
                    )
                xb = torch.cat((x, torch.ones(x.shape[0], 1)), dim=1)
                reg = torch.eye(xb.shape[1], dtype=xb.dtype) * lambda_
                reg[-1, -1] = 0.0
                theta = torch.linalg.solve(xb.T @ xb + reg, xb.T @ y)
                weights[dl, kind, h, :, : x.shape[1]] = theta[:-1].T
                bias[dl, kind, h] = theta[-1]
    metadata = MapperMetadata(
        target_layers=target_layers,
        draft_layers=draft_layers,
        target_kv_heads=kv_heads,
        draft_kv_heads=kv_heads,
        head_dim=head_dim,
        layer_selection=selection,
        lambda_=lambda_,
        content_space=content_space,
        head_mode=head_mode,
    )
    return RidgeKVMapper(metadata, weights, bias)
