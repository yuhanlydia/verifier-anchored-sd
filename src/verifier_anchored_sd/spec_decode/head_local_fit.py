"""Memory-bounded matched-head ridge fitting for verifier->draft KV transfer.

This module is a strong efficiency baseline, not an implementation claim of the
full CacheBridge method.  It keeps architecture-indexed KV heads local while using
the same content-space ridge objective as the audited full-head mapper.

For ``k`` selected verifier layers and head dimension ``d``, each solve has only
``k*d`` input features instead of ``k*H*d``.  Sufficient statistics are accumulated
with a centered Chan/Welford merge so calibration pairs can stream from CPU shards
while the post-capture fit uses an otherwise idle GPU.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass

import torch

from .cache_state import CacheState
from .target_to_draft_mapper import MapperMetadata, RidgeKVMapper

CachePair = tuple[CacheState, CacheState]
CachePairSource = Iterable[CachePair] | Callable[[], Iterable[CachePair]]


def _iterate(source: CachePairSource) -> Iterator[CachePair]:
    return iter(source() if callable(source) else source)


@dataclass
class _CenteredStats:
    count: int
    mean_x: torch.Tensor
    mean_y: torch.Tensor
    cxx: torch.Tensor
    cxy: torch.Tensor

    @classmethod
    def zeros(
        cls,
        *,
        heads: int,
        features: int,
        out_dim: int,
        device: torch.device,
    ) -> "_CenteredStats":
        return cls(
            0,
            torch.zeros(heads, features, device=device, dtype=torch.float32),
            torch.zeros(heads, out_dim, device=device, dtype=torch.float32),
            torch.zeros(heads, features, features, device=device, dtype=torch.float32),
            torch.zeros(heads, features, out_dim, device=device, dtype=torch.float32),
        )

    def update(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Merge a batch shaped ``x=[H,N,P]``, ``y=[H,N,D]``."""
        if x.ndim != 3 or y.ndim != 3 or x.shape[:2] != y.shape[:2]:
            raise ValueError("matched-head observations must be [heads, samples, features]")
        samples = x.shape[1]
        if samples == 0:
            return
        x = x.float()
        y = y.float()
        batch_mean_x = x.mean(dim=1)
        batch_mean_y = y.mean(dim=1)
        xc = x - batch_mean_x[:, None, :]
        yc = y - batch_mean_y[:, None, :]
        batch_cxx = torch.einsum("hnp,hnq->hpq", xc, xc)
        batch_cxy = torch.einsum("hnp,hnd->hpd", xc, yc)
        if self.count == 0:
            self.count = samples
            self.mean_x.copy_(batch_mean_x)
            self.mean_y.copy_(batch_mean_y)
            self.cxx.copy_(batch_cxx)
            self.cxy.copy_(batch_cxy)
            return

        old = self.count
        new = old + samples
        delta_x = batch_mean_x - self.mean_x
        delta_y = batch_mean_y - self.mean_y
        factor = float(old * samples) / float(new)
        self.cxx.add_(batch_cxx)
        self.cxx.add_(torch.einsum("hp,hq->hpq", delta_x, delta_x), alpha=factor)
        self.cxy.add_(batch_cxy)
        self.cxy.add_(torch.einsum("hp,hd->hpd", delta_x, delta_y), alpha=factor)
        self.mean_x.add_(delta_x, alpha=float(samples) / float(new))
        self.mean_y.add_(delta_y, alpha=float(samples) / float(new))
        self.count = new

    def solve(self, ridge: float) -> tuple[torch.Tensor, torch.Tensor]:
        if self.count <= self.mean_x.shape[-1]:
            raise ValueError(
                "matched-head ridge fit is underdetermined: observations must exceed feature dimension"
            )
        eye = torch.eye(self.mean_x.shape[-1], device=self.cxx.device, dtype=torch.float32)
        system = self.cxx + ridge * eye.unsqueeze(0)
        weight = torch.linalg.solve(system, self.cxy)  # [H,P,D]
        bias = self.mean_y - torch.einsum("hp,hpd->hd", self.mean_x, weight)
        return weight, bias


def _validate_pair(
    source: CacheState,
    draft: CacheState,
    *,
    target_layers: int,
    draft_layers: int,
    kv_heads: int,
    head_dim: int,
) -> None:
    if source.num_layers != target_layers or draft.num_layers != draft_layers:
        raise ValueError("calibration cache layer count does not match mapper geometry")
    if source.seq_len != draft.seq_len:
        raise ValueError("verifier/draft calibration caches must have equal sequence lengths")
    if source.kv_heads != kv_heads or draft.kv_heads != kv_heads:
        raise ValueError("matched-head fit requires equal configured KV-head counts")
    if source.head_dim != head_dim or draft.head_dim != head_dim:
        raise ValueError("calibration head dimension does not match mapper geometry")


def fit_matched_head_mapper_from_cache_pairs(
    pairs: CachePairSource,
    *,
    target_layers: int,
    draft_layers: int,
    kv_heads: int,
    head_dim: int,
    layer_selection: Sequence[Sequence[int]],
    lambda_: float = 0.01,
    accumulation_device: str | torch.device = "cuda",
    layer_block_size: int = 8,
    content_space: bool = True,
) -> RidgeKVMapper:
    """Fit independent per-head affine maps from a re-iterable cache-pair source."""
    if lambda_ < 0:
        raise ValueError("lambda_ must be non-negative")
    if layer_block_size <= 0:
        raise ValueError("layer_block_size must be positive")
    selection = [list(row) for row in layer_selection]
    if len(selection) != draft_layers or any(not row for row in selection):
        raise ValueError("one non-empty verifier-layer selection is required per draft layer")
    if any(index < 0 or index >= target_layers for row in selection for index in row):
        raise ValueError("source-layer selection contains an out-of-range layer")

    device = torch.device(accumulation_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA matched-head fitting requested but CUDA is unavailable")

    max_features = max(len(row) for row in selection) * head_dim
    weights = torch.zeros(draft_layers, 2, kv_heads, head_dim, max_features)
    bias = torch.zeros(draft_layers, 2, kv_heads, head_dim)

    reusable = pairs if callable(pairs) else list(pairs)
    total_pairs = 0
    for block_start in range(0, draft_layers, layer_block_size):
        block = range(block_start, min(block_start + layer_block_size, draft_layers))
        stats: dict[tuple[int, int], _CenteredStats] = {}
        for dl in block:
            features = len(selection[dl]) * head_dim
            for kind in (0, 1):
                stats[(dl, kind)] = _CenteredStats.zeros(
                    heads=kv_heads,
                    features=features,
                    out_dim=head_dim,
                    device=device,
                )

        block_pairs = 0
        for source_raw, draft_raw in _iterate(reusable):
            _validate_pair(
                source_raw,
                draft_raw,
                target_layers=target_layers,
                draft_layers=draft_layers,
                kv_heads=kv_heads,
                head_dim=head_dim,
            )
            source = source_raw.to(device)
            draft = draft_raw.to(device)
            if content_space:
                source = source.to_content_space()
                draft = draft.to_content_space()
            block_pairs += 1
            for dl in block:
                selected = selection[dl]
                for kind in (0, 1):
                    source_tensors = [
                        source.layers[layer].key if kind == 0 else source.layers[layer].value
                        for layer in selected
                    ]
                    x = torch.cat(source_tensors, dim=-1)  # [B,H,T,kD]
                    target_tensor = draft.layers[dl].key if kind == 0 else draft.layers[dl].value
                    # Flatten batch/token while preserving heads.
                    x = x.permute(1, 0, 2, 3).reshape(kv_heads, -1, x.shape[-1])
                    y = target_tensor.permute(1, 0, 2, 3).reshape(kv_heads, -1, head_dim)
                    stats[(dl, kind)].update(x, y)
        if block_pairs == 0:
            raise ValueError("calibration pair source was empty")
        if block_start == 0:
            total_pairs = block_pairs
        elif block_pairs != total_pairs:
            raise ValueError("calibration pair source changed across fitting passes")

        for dl in block:
            active = len(selection[dl]) * head_dim
            for kind in (0, 1):
                solved, solved_bias = stats[(dl, kind)].solve(lambda_)
                weights[dl, kind, :, :, :active] = solved.permute(0, 2, 1).cpu()
                bias[dl, kind] = solved_bias.cpu()
        del stats
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metadata = MapperMetadata(
        target_layers=target_layers,
        draft_layers=draft_layers,
        target_kv_heads=kv_heads,
        draft_kv_heads=kv_heads,
        head_dim=head_dim,
        layer_selection=selection,
        lambda_=lambda_,
        content_space=content_space,
        head_mode="matched",
    )
    return RidgeKVMapper(metadata, weights, bias)
