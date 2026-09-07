"""Distribution-level metrics and gates for verifier-to-draft pair screening."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def _normalize_probabilities(values: torch.Tensor) -> torch.Tensor:
    result = values.float()
    if not torch.isfinite(result).all():
        raise ValueError("probabilities must be finite")
    if (result < 0).any():
        raise ValueError("probabilities must be non-negative")
    totals = result.sum(dim=-1, keepdim=True)
    if (totals <= 0).any():
        raise ValueError("each probability row must have positive mass")
    return result / totals


def distribution_transfer_rows(
    native_probs: torch.Tensor,
    mapped_probs: torch.Tensor,
    next_ids: Sequence[int] | None = None,
) -> list[dict[str, float | int]]:
    """Compute native-draft versus mapped-draft metrics for independent prefixes."""
    if native_probs.shape != mapped_probs.shape or native_probs.ndim != 2:
        raise ValueError("probabilities must have identical [prefix,vocab] shapes")
    if next_ids is not None and len(next_ids) != native_probs.shape[0]:
        raise ValueError("next_ids must contain one token per probability row")
    native = _normalize_probabilities(native_probs)
    mapped = _normalize_probabilities(mapped_probs)
    rows: list[dict[str, float | int]] = []
    for index, (p, q) in enumerate(zip(native, mapped, strict=True)):
        row: dict[str, float | int] = {
            "a_transfer": float((1.0 - 0.5 * (p - q).abs().sum()).clamp(0.0, 1.0)),
            "kl_native_mapped": float(
                (p * (p.clamp_min(1e-12).log() - q.clamp_min(1e-12).log())).sum()
            ),
            "top1_agreement": int(p.argmax() == q.argmax()),
        }
        if next_ids is not None:
            token = int(next_ids[index])
            if not 0 <= token < p.numel():
                raise ValueError("next token ID lies outside the probability vocabulary")
            row["next_token_nll_delta"] = float(
                -q[token].clamp_min(1e-12).log() + p[token].clamp_min(1e-12).log()
            )
        rows.append(row)
    return rows


def classify_transfer_gate(
    ci_low: float,
    ci_high: float,
    threshold: float = 0.95,
) -> str:
    """Apply the preregistered confidence-interval pair-screen decision."""
    if not 0.0 <= ci_low <= ci_high <= 1.0:
        raise ValueError("transfer confidence interval must lie inside [0, 1]")
    if not 0.0 < threshold < 1.0:
        raise ValueError("transfer threshold must lie strictly inside (0, 1)")
    if ci_low > threshold:
        return "pass"
    if ci_high <= threshold:
        return "fail"
    return "inconclusive"


def _bootstrap_mean_interval(
    values: torch.Tensor,
    *,
    samples: int,
    seed: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randint(
        0,
        values.numel(),
        (samples, values.numel()),
        generator=generator,
    )
    means = values[indices].mean(dim=1)
    tail = (1.0 - confidence) / 2.0
    return float(torch.quantile(means, tail)), float(torch.quantile(means, 1.0 - tail))


def summarize_transfer(
    rows: Sequence[dict],
    *,
    requested: int,
    samples: int = 10000,
    seed: int = 0,
    threshold: float = 0.95,
) -> dict:
    """Aggregate pair-screen rows and withhold a gate from incomplete runs."""
    if requested <= 0:
        raise ValueError("requested row count must be positive")
    if not rows:
        raise ValueError("at least one transfer row is required")
    values = torch.tensor([float(row["a_transfer"]) for row in rows], dtype=torch.float64)
    if not torch.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("a_transfer rows must be finite and lie inside [0, 1]")
    ci_low, ci_high = _bootstrap_mean_interval(values, samples=samples, seed=seed)
    complete = len(rows) == requested
    status = (
        classify_transfer_gate(ci_low, ci_high, threshold)
        if complete
        else "incomplete"
    )
    summary = {
        "mean_a_transfer": float(values.mean()),
        "median_a_transfer": float(values.median()),
        "p05_a_transfer": float(torch.quantile(values, 0.05)),
        "min_a_transfer": float(values.min()),
        "mean_kl_native_mapped": sum(float(row["kl_native_mapped"]) for row in rows)
        / len(rows),
        "top1_agreement": sum(int(row["top1_agreement"]) for row in rows) / len(rows),
    }
    if all("next_token_nll_delta" in row for row in rows):
        summary["mean_next_token_nll_delta"] = sum(
            float(row["next_token_nll_delta"]) for row in rows
        ) / len(rows)
    return {
        "requested_rows": requested,
        "completed_rows": len(rows),
        "summary": summary,
        "gate": {
            "status": status,
            "threshold": threshold,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "confidence": 0.95,
            "bootstrap_samples": samples,
        },
    }
