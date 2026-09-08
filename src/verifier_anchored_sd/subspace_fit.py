"""Autograd and provenance helpers for student-readable KV subspace fitting."""

from __future__ import annotations

import math

import torch

from .spec_decode.cache_state import CacheState, LayerKV, RotaryFactors
from .transfer_metrics import validate_screen_inputs


def validate_subspace_fit_inputs(
    mapper_metadata: dict,
    screen_manifest: dict,
    *,
    mapper_sha256: str,
    requested_prompts: int,
) -> dict[str, int]:
    """Require a held-out verifier-distribution artifact bound to one mapper."""
    if mapper_sha256 != mapper_metadata.get("checkpoint_sha256"):
        raise RuntimeError("mapper checkpoint digest differs from mapper metadata")
    if requested_prompts <= 0:
        raise ValueError("requested_prompts must be positive")
    if screen_manifest.get("schema_version") != 3 or screen_manifest.get("role") != (
        "screen_source_with_distribution"
    ):
        raise ValueError("screen target distribution artifact must use schema version 3")
    validate_screen_inputs(mapper_metadata, screen_manifest)
    capture = screen_manifest.get("capture", {})
    captured = capture.get("count")
    if not isinstance(captured, int) or captured <= 0:
        raise ValueError("screen capture count is invalid")
    if requested_prompts > captured:
        raise ValueError("requested subspace prompts exceed captured screen rows")
    distribution = screen_manifest.get("target_distribution", {})
    vocab_size = distribution.get("vocab_size")
    if not isinstance(vocab_size, int) or vocab_size <= 0:
        raise ValueError("screen target distribution vocabulary is invalid")
    draft = mapper_metadata.get("draft_model", {})
    for field in ("layers", "kv_heads", "head_dim"):
        if not isinstance(draft.get(field), int) or int(draft[field]) <= 0:
            raise ValueError(f"mapper draft geometry lacks positive {field}")
    return {
        "draft_layers": int(draft["layers"]),
        "kv_heads": int(draft["kv_heads"]),
        "head_dim": int(draft["head_dim"]),
        "vocab_size": int(vocab_size),
        "captured_prompts": int(captured),
    }


def _detached_rotary(cache: CacheState) -> RotaryFactors | None:
    if cache.rotary is None:
        return None
    return RotaryFactors(
        cache.rotary.cos.detach().clone(),
        cache.rotary.sin.detach().clone(),
        cache.rotary.interleaved,
    )


def make_gradient_cache(cache: CacheState) -> CacheState:
    """Clone a position-space historical cache into leaf K/V tensors for autograd."""
    if cache.keys_are_content:
        raise ValueError("gradient forward requires a position-space cache")
    if cache.layers and cache.rotary is None:
        raise ValueError("gradient cache requires explicit RoPE provenance")
    return CacheState(
        (
            LayerKV(
                layer.key.detach().clone().requires_grad_(True),
                layer.value.detach().clone().requires_grad_(True),
            )
            for layer in cache.layers
        ),
        rotary=_detached_rotary(cache),
        keys_are_content=False,
    )


def _validate_single_batch(cache: CacheState) -> None:
    for layer in cache.layers:
        if layer.key.shape[0] != 1:
            raise ValueError("subspace fitting currently requires cache batch size one")


def cache_content_rows(cache: CacheState) -> torch.Tensor:
    """Return cache rows as ``[layers, K/V, heads, tokens, head_dim]``."""
    if not cache.layers:
        raise ValueError("cache must contain at least one layer")
    _validate_single_batch(cache)
    content = cache if cache.keys_are_content else cache.to_content_space()
    layers = []
    for layer in content.layers:
        layers.append(torch.stack((layer.key[0], layer.value[0]), dim=0))
    rows = torch.stack(layers, dim=0)
    if not torch.isfinite(rows).all():
        raise RuntimeError("cache content rows contain non-finite values")
    return rows


def cache_gradient_rows(cache: CacheState) -> torch.Tensor:
    """Return leaf-cache gradients in K-content/V space.

    A position-space key is ``k_pos = R k_content``.  Because RoPE is orthogonal,
    the content-space gradient is ``R^T g_pos``, implemented by inverse RoPE.
    """
    if not cache.layers:
        raise ValueError("gradient cache must contain at least one layer")
    if cache.keys_are_content or cache.rotary is None:
        raise ValueError("gradient cache must be position-space with RoPE provenance")
    _validate_single_batch(cache)
    layers = []
    for layer in cache.layers:
        if layer.key.grad is None or layer.value.grad is None:
            raise RuntimeError("gradient cache tensors have no gradients; backward was not run")
        key_grad = cache.rotary.apply(layer.key.grad, inverse=True)
        value_grad = layer.value.grad
        if not torch.isfinite(key_grad).all() or not torch.isfinite(value_grad).all():
            raise RuntimeError("cache gradient contains non-finite values")
        layers.append(torch.stack((key_grad[0], value_grad[0]), dim=0))
    return torch.stack(layers, dim=0)


def teacher_delta_rows(native: CacheState, mapped: CacheState) -> torch.Tensor:
    """Return ``mapped-native`` historical KV in common K-content/V coordinates."""
    if (
        native.num_layers != mapped.num_layers
        or native.kv_heads != mapped.kv_heads
        or native.head_dim != mapped.head_dim
        or native.seq_len != mapped.seq_len
    ):
        raise ValueError("native and mapped caches must have identical draft geometry")
    native_rows = cache_content_rows(native)
    mapped_rows = cache_content_rows(mapped)
    delta = mapped_rows - native_rows
    if not torch.isfinite(delta).all():
        raise RuntimeError("teacher delta contains non-finite values")
    return delta


def gradient_norms(rows: torch.Tensor) -> torch.Tensor:
    """Return L2 norms over token/head-dimension rows as ``[layers, 2, heads]``."""
    if rows.ndim != 5 or rows.shape[1] != 2:
        raise ValueError("gradient rows must be [layers,2,heads,tokens,head_dim]")
    values = rows.detach().float()
    if not torch.isfinite(values).all():
        raise RuntimeError("gradient rows must be finite")
    return torch.linalg.vector_norm(values, dim=(-2, -1)).cpu()


def assert_gradient_coverage(norm_sums: torch.Tensor, *, prefixes: int) -> None:
    """Fail a gradient smoke unless every layer/KV-kind/head receives signal."""
    if prefixes <= 0:
        raise ValueError("prefixes must be positive")
    if norm_sums.ndim != 3 or norm_sums.shape[1] != 2:
        raise ValueError("gradient norm summary must be [layers,2,heads]")
    if not torch.isfinite(norm_sums).all():
        raise RuntimeError("gradient norm summary must be finite")
    zero = torch.nonzero(norm_sums <= 0, as_tuple=False)
    if zero.numel():
        locations = [tuple(int(x) for x in row.tolist()) for row in zero[:16]]
        raise RuntimeError(
            f"zero gradient detected after {prefixes} prefixes at layer/kind/head={locations}"
        )


def target_kl_loss(logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    """Compute ``KL(p_target || q_student)`` at the final next-token distribution."""
    if logits.ndim == 3:
        if logits.shape[0] != 1:
            raise ValueError("target KL fitting requires batch size one")
        student_logits = logits[0, -1]
    elif logits.ndim == 2:
        student_logits = logits[-1]
    elif logits.ndim == 1:
        student_logits = logits
    else:
        raise ValueError("logits must be [vocab], [tokens,vocab], or [1,tokens,vocab]")
    target = target_probs.to(device=student_logits.device, dtype=torch.float32)
    if target.ndim != 1 or target.numel() != student_logits.numel():
        raise ValueError("target probability vocabulary differs from student logits")
    if not torch.isfinite(target).all() or (target < 0).any():
        raise ValueError("target probabilities must be finite and non-negative")
    total = target.sum()
    if not torch.allclose(total, torch.tensor(1.0, device=target.device), atol=1e-5, rtol=1e-5):
        raise ValueError("target probabilities must sum to one")
    log_target = target.clamp_min(1e-12).log()
    log_student = torch.log_softmax(student_logits.float(), dim=-1)
    loss = torch.sum(target * (log_target - log_student))
    if not torch.isfinite(loss):
        raise RuntimeError("target KL loss is non-finite")
    return loss
