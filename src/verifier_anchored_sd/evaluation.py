"""Deterministic evaluation helpers shared by benchmark scripts."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch


def acceptance_methods() -> dict[str, dict[str, str]]:
    """Return the preregistered E2 method matrix in stable execution order."""
    return {
        "native_sd": {"init_mode": "native", "refresh_policy": "none"},
        "legacy_mapped_init_only": {
            "init_mode": "legacy_mapped",
            "refresh_policy": "none",
        },
        "mapped_init_only": {
            "init_mode": "mapped_native_frontier",
            "refresh_policy": "none",
        },
        "legacy_full_refresh": {
            "init_mode": "legacy_mapped",
            "refresh_policy": "full",
        },
        "mapped_accepted_only": {
            "init_mode": "mapped_native_frontier",
            "refresh_policy": "accepted_only",
        },
    }


def classify_mapper_retention(retention: float) -> str:
    """Classify mapped/native expected-MAL retention before refresh inference."""
    if not math.isfinite(retention) or retention < 0:
        raise ValueError("mapper retention must be finite and non-negative")
    if retention >= 0.90:
        return "confirmatory"
    if retention >= 0.85:
        return "exploratory"
    return "reject"


def classify_refresh_delta(ci_low: float, ci_high: float) -> str:
    """Apply the preregistered sign decision to a paired refresh interval."""
    if not math.isfinite(ci_low) or not math.isfinite(ci_high) or ci_low > ci_high:
        raise ValueError("refresh confidence interval must be finite and ordered")
    if ci_low > 0:
        return "support"
    if ci_high < 0:
        return "stop"
    return "inconclusive"


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
    proposal_token_ids: Sequence[int] | torch.Tensor,
) -> torch.Tensor:
    """Return expected accepted length conditional on sampled proposal tokens.

    For sampled proposal token ``x_i``, exact speculative decoding accepts with
    probability ``min(1, p_i(x_i) / q_i(x_i))``. Reaching position ``j`` requires
    every earlier realized proposal to be accepted, so the conditional expectation
    is ``sum_j prod_{i<=j} min(1, p_i(x_i)/q_i(x_i))``. Averaging this quantity over
    independently sampled proposal paths gives a Monte Carlo estimate of marginal
    expected accepted length.

    Inputs may be ``[gamma, vocab]`` or ``[batch, gamma, vocab]``.  The return value
    is scalar for the former and ``[batch]`` for the latter.
    """
    if target_probs.shape != draft_probs.shape:
        raise ValueError("target and draft probability tensors must have identical shapes")
    if target_probs.ndim not in {2, 3}:
        raise ValueError("probabilities must be [gamma,vocab] or [batch,gamma,vocab]")
    token_ids = torch.as_tensor(proposal_token_ids, dtype=torch.long, device=target_probs.device)
    expected_shape = target_probs.shape[:-1]
    valid_shape = (
        target_probs.ndim == 2 and token_ids.shape == (expected_shape[0],)
    ) or (target_probs.ndim == 3 and token_ids.shape == expected_shape)
    if not valid_shape:
        raise ValueError("proposal token IDs must match the probability block dimensions")
    if ((token_ids < 0) | (token_ids >= target_probs.shape[-1])).any():
        raise ValueError("proposal token ID lies outside the probability vocabulary")
    p = target_probs.float()
    q = draft_probs.float()
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    q = q / q.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    selected = token_ids.unsqueeze(-1)
    p_token = p.gather(-1, selected).squeeze(-1)
    q_token = q.gather(-1, selected).squeeze(-1)
    if (q_token <= 0).any():
        raise ValueError("sampled proposal token must have positive draft probability")
    alpha = (p_token / q_token).clamp(max=1.0)
    block_dim = 0 if alpha.ndim == 1 else 1
    return torch.cumprod(alpha, dim=block_dim).sum(dim=block_dim)


def paired_bootstrap_mean_difference(
    a: Sequence[float],
    b: Sequence[float],
    *,
    samples: int = 10000,
    seed: int = 0,
    confidence: float = 0.95,
    cluster_ids: Sequence[int | str] | None = None,
) -> dict[str, float]:
    """Paired bootstrap CI for ``mean(a-b)`` without a NumPy dependency."""
    if len(a) != len(b) or not a:
        raise ValueError("paired samples must be non-empty and have identical lengths")
    if samples <= 0:
        raise ValueError("samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    delta = torch.tensor(a, dtype=torch.float64) - torch.tensor(b, dtype=torch.float64)
    if cluster_ids is None:
        cluster_ids = list(range(delta.numel()))
    if len(cluster_ids) != delta.numel():
        raise ValueError("cluster_ids must contain one ID per paired row")
    ordered = list(dict.fromkeys(cluster_ids))
    cluster_index = {cluster: index for index, cluster in enumerate(ordered)}
    sums = torch.zeros(len(ordered), dtype=torch.float64)
    counts = torch.zeros(len(ordered), dtype=torch.float64)
    for value, cluster in zip(delta, cluster_ids, strict=True):
        index = cluster_index[cluster]
        sums[index] += value
        counts[index] += 1
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randint(
        0,
        len(ordered),
        (samples, len(ordered)),
        generator=generator,
    )
    means = sums[indices].sum(dim=1) / counts[indices].sum(dim=1)
    tail = (1.0 - confidence) / 2.0
    return {
        "mean_difference": float(delta.mean()),
        "ci_low": float(torch.quantile(means, tail)),
        "ci_high": float(torch.quantile(means, 1.0 - tail)),
        "confidence": float(confidence),
        "bootstrap_samples": float(samples),
        "pairs": float(delta.numel()),
        "independent_clusters": float(len(ordered)),
    }
