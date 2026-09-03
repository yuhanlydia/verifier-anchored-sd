"""Deterministic evaluation helpers shared by benchmark scripts."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def block_bucket_overlap(*, cursor: int, emitted: int, lo: int, hi: int) -> int:
    """Count emitted output positions that fall in the inclusive bucket [lo, hi].

    ``cursor`` is the number of output tokens emitted before the block, so the
    block occupies one-indexed positions ``cursor+1 .. cursor+emitted``.
    """
    if cursor < 0 or emitted < 0:
        raise ValueError("cursor and emitted must be non-negative")
    if lo < 1 or hi < lo:
        raise ValueError("bucket must be a non-empty positive inclusive interval")
    if emitted == 0:
        return 0
    start = max(cursor + 1, lo)
    end = min(cursor + emitted, hi)
    return max(0, end - start + 1)


def expected_accepted_length(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
) -> torch.Tensor:
    """Return the exact expected accepted proposal length for a fixed SD block.

    For each proposal position ``i``, rejection sampling accepts with expected mass
    ``alpha_i = sum_v min(p_i(v), q_i(v)) = 1 - TV(p_i, q_i)``.  Reaching position
    ``j`` requires every earlier proposal to be accepted, therefore
    ``E[L] = sum_j prod_{i<=j} alpha_i``.

    Inputs may be ``[gamma, vocab]`` or ``[batch, gamma, vocab]``.  The return value
    is scalar for the former and ``[batch]`` for the latter.
    """
    if target_probs.shape != draft_probs.shape:
        raise ValueError("target and draft probability tensors must have identical shapes")
    if target_probs.ndim not in {2, 3}:
        raise ValueError("probabilities must be [gamma,vocab] or [batch,gamma,vocab]")
    p = target_probs.float()
    q = draft_probs.float()
    # Be robust to tiny numerical normalization error from serialized/logit paths.
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    q = q / q.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    alpha = (1.0 - 0.5 * (p - q).abs().sum(dim=-1)).clamp(0.0, 1.0)
    block_dim = 0 if alpha.ndim == 1 else 1
    return torch.cumprod(alpha, dim=block_dim).sum(dim=block_dim)


def paired_bootstrap_mean_difference(
    a: Sequence[float],
    b: Sequence[float],
    *,
    samples: int = 10000,
    seed: int = 0,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Paired bootstrap CI for ``mean(a-b)`` without a NumPy dependency."""
    if len(a) != len(b) or not a:
        raise ValueError("paired samples must be non-empty and have identical lengths")
    if samples <= 0:
        raise ValueError("samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    delta = torch.tensor(a, dtype=torch.float64) - torch.tensor(b, dtype=torch.float64)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randint(
        0,
        delta.numel(),
        (samples, delta.numel()),
        generator=generator,
    )
    means = delta[indices].mean(dim=1)
    tail = (1.0 - confidence) / 2.0
    return {
        "mean_difference": float(delta.mean()),
        "ci_low": float(torch.quantile(means, tail)),
        "ci_high": float(torch.quantile(means, 1.0 - tail)),
        "confidence": float(confidence),
        "bootstrap_samples": float(samples),
        "pairs": float(delta.numel()),
    }
