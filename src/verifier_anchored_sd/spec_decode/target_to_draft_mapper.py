"""Target-to-draft KV ridge mapping and mergeable low-rank residuals."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from .cache_state import CacheState, LayerKV


def default_layer_selection(target_layers: int, draft_layers: int, k: int = 8) -> list[list[int]]:
    """Choose evenly spaced source layers, excluding duplicate indices."""
    if target_layers < 1 or draft_layers < 1 or k < 1:
        raise ValueError("layer counts and k must be positive")
    k = min(k, target_layers)
    result = []
    for d in range(draft_layers):
        center = round((d + 0.5) * target_layers / draft_layers - 0.5)
        offsets = sorted(range(target_layers), key=lambda i: (abs(i - center), i))
        result.append(sorted(offsets[:k]))
    return result


@dataclass
class MapperMetadata:
    target_layers: int
    draft_layers: int
    kv_heads: int
    head_dim: int
    layer_selection: list[list[int]]
    lambda_: float


class RidgeKVMapper:
    """Per draft-layer/head affine mapper with an optional mergeable LoRA-like residual."""

    def __init__(self, metadata: MapperMetadata, weights: torch.Tensor, bias: torch.Tensor) -> None:
        # weights: [draft_layers, 2, heads, out_dim, in_dim]
        # bias:    [draft_layers, 2, heads, out_dim]
        expected = (metadata.draft_layers, 2, metadata.kv_heads, metadata.head_dim)
        if tuple(weights.shape[:3]) != expected[:3] or weights.shape[3] != metadata.head_dim:
            raise ValueError(f"unexpected mapper weight shape {weights.shape}")
        if tuple(bias.shape) != expected:
            raise ValueError(f"unexpected mapper bias shape {bias.shape}")
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

    def _map_kind(self, x: torch.Tensor, layer: int, kind: int) -> torch.Tensor:
        # x: [B, H*k, T, D], output: [B, H, T, D]
        output_dtype = x.dtype
        x = x.to(self.weights.dtype)
        b, _, t, d = x.shape
        h = self.metadata.kv_heads
        x = x.reshape(b, h, -1, t, d).permute(0, 1, 3, 2, 4).reshape(b, h, t, -1)
        w = self.weights[layer, kind]
        y = torch.einsum("bhti,hoi->bhto", x, w)
        y = y + self.bias[layer, kind].view(1, h, 1, -1)
        if self.u is not None and self.v is not None:
            u = self.u[layer, kind]
            v = self.v[layer, kind]
            y = y + self.gate.to(y.dtype) * torch.einsum("bhti,hri,hor->bhto", x, v, u)
        return y.to(output_dtype)

    def map(self, target: CacheState) -> CacheState:
        if target.num_layers != self.metadata.target_layers:
            raise ValueError("target cache layer count does not match mapper")
        result = []
        for dl, selected in enumerate(self.metadata.layer_selection):
            keys = torch.cat([target.layers[i].key for i in selected], dim=1)
            values = torch.cat([target.layers[i].value for i in selected], dim=1)
            result.append(LayerKV(self._map_kind(keys, dl, 0), self._map_kind(values, dl, 1)))
        return CacheState(result)

    def add_low_rank(self, rank: int = 8) -> None:
        if rank < 1:
            raise ValueError("rank must be positive")
        shape = (self.metadata.draft_layers, 2, self.metadata.kv_heads)
        self.u = nn.Parameter(torch.randn(*shape, self.weights.shape[-2], rank, device=self.device, dtype=self.weights.dtype) * 1e-3)
        self.v = nn.Parameter(torch.randn(*shape, rank, self.weights.shape[-1], device=self.device, dtype=self.weights.dtype) * 1e-3)
        self.gate = nn.Parameter(torch.tensor(0.0, device=self.device, dtype=self.weights.dtype))

    def merge_residual(self) -> RidgeKVMapper:
        if self.u is None or self.v is None:
            return self
        delta = torch.einsum("...or,...ri->...oi", self.u, self.v)
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
            "weights": self.weights.cpu(), "bias": self.bias.cpu(),
            "metadata": self.metadata.__dict__, "gate": self.gate.cpu(),
            "u": None if self.u is None else self.u.cpu(),
            "v": None if self.v is None else self.v.cpu(),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping) -> RidgeKVMapper:
        metadata = MapperMetadata(**state["metadata"])
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
    def load(cls, path: str | Path, map_location: str | torch.device = "cpu") -> RidgeKVMapper:
        return cls.from_state_dict(torch.load(path, map_location=map_location, weights_only=False))


def fit_ridge_mapper(
    observations: Mapping[tuple[int, int, str], tuple[torch.Tensor, torch.Tensor]],
    *,
    target_layers: int,
    draft_layers: int,
    kv_heads: int,
    head_dim: int,
    layer_selection: Sequence[Sequence[int]] | None = None,
    lambda_: float = 0.01,
) -> RidgeKVMapper:
    """Fit independent affine ridge regressions.

    ``observations[(draft_layer, draft_head, kind)]`` contains ``(X, Y)`` with
    ``X=[N, k*head_dim]`` and ``Y=[N, head_dim]``.  Keeping this interface model
    agnostic makes calibration reproducible and permits fitting from saved activations.
    """
    if lambda_ < 0:
        raise ValueError("lambda_ must be non-negative")
    selection = [list(x) for x in (layer_selection or default_layer_selection(target_layers, draft_layers))]
    if len(selection) != draft_layers:
        raise ValueError("one source-layer selection is required per draft layer")
    w = torch.zeros(draft_layers, 2, kv_heads, head_dim, max(len(x) for x in selection) * head_dim)
    b = torch.zeros(draft_layers, 2, kv_heads, head_dim)
    for dl in range(draft_layers):
        k = len(selection[dl])
        for h in range(kv_heads):
            for kind, name in enumerate(("k", "v")):
                x, y = observations[(dl, h, name)]
                x, y = x.float(), y.float()
                if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0] or y.shape[1] != head_dim:
                    raise ValueError(f"invalid observation shape for {(dl, h, name)}")
                if x.shape[1] != k * head_dim:
                    raise ValueError(f"expected {k * head_dim} input features, got {x.shape[1]}")
                xb = torch.cat((x, torch.ones(x.shape[0], 1)), dim=1)
                reg = torch.eye(xb.shape[1], dtype=xb.dtype) * lambda_
                reg[-1, -1] = 0.0
                theta = torch.linalg.solve(xb.T @ xb + reg, xb.T @ y)
                w[dl, kind, h, :, : x.shape[1]] = theta[:-1].T
                b[dl, kind, h] = theta[-1]
    metadata = MapperMetadata(target_layers, draft_layers, kv_heads, head_dim, selection, lambda_)
    return RidgeKVMapper(metadata, w, b)


def fit_ridge_mapper_from_cache_pairs(
    pairs,
    *,
    target_layers: int,
    draft_layers: int,
    kv_heads: int,
    head_dim: int,
    layer_selection: Sequence[Sequence[int]] | None = None,
    lambda_: float = 0.01,
    stride: int = 4,
) -> RidgeKVMapper:
    """Fit from a one-pass iterable of ``(target_cache, native_draft_cache)``.

    Only normal equations are retained, so a 128K-observation calibration does not
    accumulate all activations in RAM.  This is the intended E0 path.
    """
    selection = [list(x) for x in (layer_selection or default_layer_selection(target_layers, draft_layers))]
    max_in = max(len(x) for x in selection) * head_dim
    stats: dict[tuple[int, int, str], list[torch.Tensor]] = {}
    for dl in range(draft_layers):
        for h in range(kv_heads):
            for name in ("k", "v"):
                stats[(dl, h, name)] = [
                    torch.zeros(max_in + 1, max_in + 1, dtype=torch.float32),
                    torch.zeros(max_in + 1, head_dim, dtype=torch.float32),
                ]
    count = 0
    for target, native in pairs:
        if target.seq_len != native.seq_len:
            raise ValueError("target/native calibration caches must have equal sequence length")
        idx = slice(0, target.seq_len, stride)
        for dl, selected in enumerate(selection):
            for h in range(kv_heads):
                for kind, name in ((0, "k"), (1, "v")):
                    source = [target.layers[i].key if kind == 0 else target.layers[i].value for i in selected]
                    x = torch.cat([z[:, h, idx, :].reshape(-1, head_dim) for z in source], dim=1).float()
                    y = (native.layers[dl].key if kind == 0 else native.layers[dl].value)[:, h, idx, :].reshape(-1, head_dim).float()
                    xb = torch.cat((x, torch.ones(x.shape[0], 1, dtype=torch.float32)), dim=1)
                    stats[(dl, h, name)][0].add_(xb.T @ xb)
                    stats[(dl, h, name)][1].add_(xb.T @ y)
        count += 1
    if count == 0:
        raise ValueError("calibration pair iterable was empty")
    for key, (xtx, xty) in stats.items():
        xtx, xty = xtx.double(), xty.double()
        reg = torch.eye(max_in + 1, dtype=torch.float64) * lambda_
        reg[-1, -1] = 0.0
        theta = torch.linalg.solve(xtx + reg, xty)
        dl, h, name = key
        k = len(selection[dl])
        # Store solved parameters temporarily in a second compact structure.
        stats[key] = [theta, torch.tensor(0.0)]
    weights = torch.zeros(draft_layers, 2, kv_heads, head_dim, max_in)
    bias = torch.zeros(draft_layers, 2, kv_heads, head_dim)
    for (dl, h, name), (theta, _) in stats.items():
        kind = 0 if name == "k" else 1
        k = len(selection[dl]) * head_dim
        weights[dl, kind, h, :, :k] = theta[:-1].T.float()
        bias[dl, kind, h] = theta[-1].float()
    metadata = MapperMetadata(target_layers, draft_layers, kv_heads, head_dim, selection, lambda_)
    return RidgeKVMapper(metadata, weights, bias)
