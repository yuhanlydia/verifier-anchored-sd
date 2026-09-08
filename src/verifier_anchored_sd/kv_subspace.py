"""Student-readable KV subspace artifacts and cache interventions.

Subspace bases are stored per draft layer, K/V kind, KV head and head dimension.
Keys are always projected in RoPE-free content space; values are projected in their
native value space.  Interventions never touch the causal frontier token: callers
apply them to historical cache only, then run the newest token through the draft
model natively.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch

from .spec_decode.cache_state import CacheState, LayerKV

InterventionMode = Literal[
    "mapped_soft",
    "delta_projected",
    "delta_shrink",
    "orthogonal",
]


@dataclass(frozen=True)
class InterventionSpec:
    """One frozen historical-KV intervention configuration."""

    mode: InterventionMode
    family: str
    rank: int
    beta: float | None = None
    alpha: float | None = None

    def validate(self) -> None:
        if self.mode not in {
            "mapped_soft",
            "delta_projected",
            "delta_shrink",
            "orthogonal",
        }:
            raise ValueError(f"unsupported intervention mode: {self.mode}")
        if not self.family:
            raise ValueError("basis family must be non-empty")
        if self.rank <= 0:
            raise ValueError("subspace rank must be positive")
        if self.mode in {"mapped_soft", "delta_shrink"}:
            if self.beta is None or not math.isfinite(self.beta) or not 0.0 <= self.beta <= 1.0:
                raise ValueError("beta must be finite and lie in [0, 1]")
        elif self.beta is not None:
            raise ValueError(f"beta is not used by intervention mode {self.mode}")
        if self.mode == "delta_projected":
            if self.alpha is None or not math.isfinite(self.alpha) or self.alpha < 0.0:
                raise ValueError("alpha must be finite and non-negative")
        elif self.alpha is not None:
            raise ValueError(f"alpha is not used by intervention mode {self.mode}")


@dataclass
class SubspaceBasisArtifact:
    """In-memory collection of ordered orthonormal KV subspace bases.

    ``bases[family]`` has shape ``[layers, 2, heads, head_dim, max_rank]`` where
    kind 0 is K and kind 1 is V. ``eigenvalues[family]`` has the same leading
    ``[layers, 2, heads]`` axes and a final ``max_rank`` axis. ``valid_ranks``
    allows signed families such as positive-benefit eigenspaces to expose fewer
    than ``max_rank`` usable directions for an individual layer/head/kind.
    """

    metadata: dict
    bases: dict[str, torch.Tensor]
    eigenvalues: dict[str, torch.Tensor]
    valid_ranks: dict[str, torch.Tensor]

    @property
    def draft_layers(self) -> int:
        return int(self.metadata["draft_layers"])

    @property
    def kv_heads(self) -> int:
        return int(self.metadata["kv_heads"])

    @property
    def head_dim(self) -> int:
        return int(self.metadata["head_dim"])

    @property
    def max_rank(self) -> int:
        return int(self.metadata["max_rank"])

    def validate(self) -> None:
        if self.draft_layers <= 0 or self.kv_heads <= 0 or self.head_dim <= 0:
            raise ValueError("subspace geometry must be positive")
        if not 0 < self.max_rank <= self.head_dim:
            raise ValueError("max_rank must lie inside the head dimension")
        families = set(self.bases)
        if not families or families != set(self.eigenvalues) or families != set(self.valid_ranks):
            raise ValueError("basis/eigenvalue/valid-rank families must match and be non-empty")
        expected_basis = (
            self.draft_layers,
            2,
            self.kv_heads,
            self.head_dim,
            self.max_rank,
        )
        expected_values = (self.draft_layers, 2, self.kv_heads, self.max_rank)
        expected_ranks = (self.draft_layers, 2, self.kv_heads)
        eye_cache: dict[int, torch.Tensor] = {}
        for family in sorted(families):
            basis = self.bases[family]
            values = self.eigenvalues[family]
            ranks = self.valid_ranks[family]
            if tuple(basis.shape) != expected_basis:
                raise ValueError(f"basis family {family!r} has shape {tuple(basis.shape)}, expected {expected_basis}")
            if tuple(values.shape) != expected_values:
                raise ValueError(f"eigenvalues for {family!r} have an invalid shape")
            if tuple(ranks.shape) != expected_ranks:
                raise ValueError(f"valid ranks for {family!r} have an invalid shape")
            if not torch.isfinite(basis).all() or not torch.isfinite(values).all():
                raise ValueError(f"basis family {family!r} must be finite")
            if ranks.dtype not in {
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            }:
                raise ValueError("valid ranks must use an integer dtype")
            if (ranks < 0).any() or (ranks > self.max_rank).any():
                raise ValueError("valid ranks must lie in [0, max_rank]")
            for layer in range(self.draft_layers):
                for kind in range(2):
                    for head in range(self.kv_heads):
                        rank = int(ranks[layer, kind, head])
                        if rank == 0:
                            continue
                        u = basis[layer, kind, head, :, :rank].float()
                        eye = eye_cache.setdefault(rank, torch.eye(rank, dtype=torch.float32))
                        gram = u.T @ u
                        if not torch.allclose(gram.cpu(), eye, atol=1e-4, rtol=1e-4):
                            raise ValueError(
                                f"basis family {family!r} is not orthonormal at "
                                f"layer={layer}, kind={kind}, head={head}"
                            )

    def effective_rank(
        self,
        family: str,
        *,
        requested_rank: int,
        layer: int,
        kind: int,
        head: int,
    ) -> int:
        if family not in self.bases:
            raise KeyError(f"unknown basis family: {family}")
        if requested_rank <= 0:
            raise ValueError("requested_rank must be positive")
        if not 0 <= layer < self.draft_layers or kind not in {0, 1} or not 0 <= head < self.kv_heads:
            raise IndexError("subspace basis index is outside artifact geometry")
        return min(requested_rank, int(self.valid_ranks[family][layer, kind, head]))


def deterministic_random_basis(
    *,
    layers: int,
    kinds: int,
    heads: int,
    head_dim: int,
    max_rank: int,
    seed: int,
) -> torch.Tensor:
    """Return deterministic rank-matched random orthonormal bases on CPU."""
    if min(layers, kinds, heads, head_dim, max_rank) <= 0:
        raise ValueError("random basis dimensions must be positive")
    if max_rank > head_dim:
        raise ValueError("max_rank cannot exceed head_dim")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    result = torch.empty(layers, kinds, heads, head_dim, max_rank, dtype=torch.float32)
    for layer in range(layers):
        for kind in range(kinds):
            for head in range(heads):
                matrix = torch.randn(head_dim, max_rank, generator=generator)
                q, _ = torch.linalg.qr(matrix, mode="reduced")
                # QR signs are deterministic for a fixed PyTorch build, but make the
                # convention explicit so serialization / comparisons are stable.
                dominant = q.abs().argmax(dim=0)
                signs = torch.sign(q[dominant, torch.arange(max_rank)])
                signs = torch.where(signs == 0, torch.ones_like(signs), signs)
                result[layer, kind, head] = q * signs
    return result


def _validate_cache_pair(native: CacheState, mapped: CacheState, artifact: SubspaceBasisArtifact) -> None:
    if native.num_layers != mapped.num_layers or native.num_layers != artifact.draft_layers:
        raise ValueError("native/mapped/subspace layer counts differ")
    if native.kv_heads != mapped.kv_heads or native.kv_heads != artifact.kv_heads:
        raise ValueError("native/mapped/subspace KV-head counts differ")
    if native.head_dim != mapped.head_dim or native.head_dim != artifact.head_dim:
        raise ValueError("native/mapped/subspace head dimensions differ")
    if native.seq_len != mapped.seq_len:
        raise ValueError("native and mapped historical cache lengths differ")
    if native.keys_are_content or mapped.keys_are_content:
        raise ValueError("runtime intervention expects position-space caches with RoPE provenance")
    if native.rotary is None or mapped.rotary is None:
        raise ValueError("runtime intervention requires draft RoPE factors")
    if native.rotary.interleaved != mapped.rotary.interleaved:
        raise ValueError("native and mapped caches use different RoPE conventions")
    if native.rotary.cos.shape != mapped.rotary.cos.shape or native.rotary.sin.shape != mapped.rotary.sin.shape:
        raise ValueError("native and mapped caches use different RoPE shapes")
    if not torch.allclose(native.rotary.cos.float().cpu(), mapped.rotary.cos.float().cpu(), atol=1e-6, rtol=1e-6):
        raise ValueError("native and mapped caches use different RoPE cosine factors")
    if not torch.allclose(native.rotary.sin.float().cpu(), mapped.rotary.sin.float().cpu(), atol=1e-6, rtol=1e-6):
        raise ValueError("native and mapped caches use different RoPE sine factors")


def _project_kind(
    tensor: torch.Tensor,
    artifact: SubspaceBasisArtifact,
    family: str,
    rank: int,
    *,
    layer: int,
    kind: int,
) -> torch.Tensor:
    if tensor.ndim != 4 or tensor.shape[1] != artifact.kv_heads or tensor.shape[-1] != artifact.head_dim:
        raise ValueError("cache tensor does not match subspace geometry")
    output = torch.zeros_like(tensor)
    for head in range(artifact.kv_heads):
        effective = artifact.effective_rank(
            family,
            requested_rank=rank,
            layer=layer,
            kind=kind,
            head=head,
        )
        if effective == 0:
            continue
        u = artifact.bases[family][layer, kind, head, :, :effective].to(
            device=tensor.device, dtype=torch.float32
        )
        x = tensor[:, head].float()
        coeff = torch.einsum("btd,dr->btr", x, u)
        projected = torch.einsum("btr,dr->btd", coeff, u)
        output[:, head] = projected.to(tensor.dtype)
    return output


def apply_intervention(
    native: CacheState,
    mapped: CacheState,
    artifact: SubspaceBasisArtifact,
    spec: InterventionSpec,
) -> CacheState:
    """Apply one historical-cache intervention and return position-space draft KV."""
    spec.validate()
    artifact.validate()
    _validate_cache_pair(native, mapped, artifact)
    if spec.family not in artifact.bases:
        raise KeyError(f"unknown basis family: {spec.family}")

    # Preserve exact endpoint identities instead of introducing avoidable
    # inverse-RoPE/re-RoPE roundoff when the requested method is the baseline.
    if spec.mode == "mapped_soft" and spec.beta == 1.0:
        return mapped.clone()
    if spec.mode == "delta_shrink" and spec.beta == 1.0:
        return mapped.clone()
    if spec.mode == "delta_projected" and spec.alpha == 0.0:
        return native.clone()

    native_content = native.to_content_space()
    mapped_content = mapped.to_content_space()
    result_layers: list[LayerKV] = []
    for layer_index, (native_layer, mapped_layer) in enumerate(
        zip(native_content.layers, mapped_content.layers, strict=True)
    ):
        outputs = []
        for kind, (native_tensor, mapped_tensor) in enumerate(
            ((native_layer.key, mapped_layer.key), (native_layer.value, mapped_layer.value))
        ):
            if spec.mode == "mapped_soft":
                projected = _project_kind(
                    mapped_tensor,
                    artifact,
                    spec.family,
                    spec.rank,
                    layer=layer_index,
                    kind=kind,
                )
                output = projected + float(spec.beta) * (mapped_tensor - projected)
            elif spec.mode == "orthogonal":
                projected = _project_kind(
                    mapped_tensor,
                    artifact,
                    spec.family,
                    spec.rank,
                    layer=layer_index,
                    kind=kind,
                )
                output = mapped_tensor - projected
            else:
                delta = mapped_tensor - native_tensor
                projected_delta = _project_kind(
                    delta,
                    artifact,
                    spec.family,
                    spec.rank,
                    layer=layer_index,
                    kind=kind,
                )
                if spec.mode == "delta_projected":
                    output = native_tensor + float(spec.alpha) * projected_delta
                elif spec.mode == "delta_shrink":
                    output = (
                        native_tensor
                        + projected_delta
                        + float(spec.beta) * (delta - projected_delta)
                    )
                else:  # guarded by InterventionSpec.validate
                    raise AssertionError("unreachable intervention mode")
            if not torch.isfinite(output).all():
                raise RuntimeError("subspace intervention produced non-finite KV")
            outputs.append(output)
        result_layers.append(LayerKV(outputs[0], outputs[1]))

    content = CacheState(
        result_layers,
        rotary=mapped_content.rotary,
        keys_are_content=True,
    )
    assert mapped.rotary is not None
    return content.apply_rotary(mapped.rotary)
