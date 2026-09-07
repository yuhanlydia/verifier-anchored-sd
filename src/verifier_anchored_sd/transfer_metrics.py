"""Distribution-level metrics and gates for verifier-to-draft pair screening."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .experiment_artifacts import validate_no_row_overlap


def _normalize_probabilities(values: torch.Tensor) -> torch.Tensor:
    result = values.float()
    if not torch.isfinite(result).all():
        raise ValueError("probabilities must be finite")
    if (result < 0).any():
        raise ValueError("probabilities must be non-negative")
    totals = result.sum(dim=-1, keepdim=True)
    if (totals <= 0).any():
        raise ValueError("each probability row must have positive mass")
    if not torch.allclose(totals, torch.ones_like(totals), atol=1e-5, rtol=1e-5):
        raise ValueError("each probability row must sum to one")
    return result


def _overlap(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (1.0 - 0.5 * (left - right).abs().sum()).clamp(0.0, 1.0)


def _kl(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (
        left * (left.clamp_min(1e-12).log() - right.clamp_min(1e-12).log())
    ).sum()


def distribution_transfer_rows(
    native_probs: torch.Tensor,
    mapped_probs: torch.Tensor,
    next_ids: Sequence[int] | None = None,
) -> list[dict[str, float | int]]:
    """Compute native-draft versus mapped-draft metrics for prefix rows."""
    if native_probs.shape != mapped_probs.shape or native_probs.ndim != 2:
        raise ValueError("probabilities must have identical [prefix,vocab] shapes")
    if next_ids is not None and len(next_ids) != native_probs.shape[0]:
        raise ValueError("next_ids must contain one token per probability row")
    native = _normalize_probabilities(native_probs)
    mapped = _normalize_probabilities(mapped_probs)
    rows: list[dict[str, float | int]] = []
    for index, (p, q) in enumerate(zip(native, mapped, strict=True)):
        row: dict[str, float | int] = {
            "a_transfer": float(_overlap(p, q)),
            "kl_native_mapped": float(_kl(p, q)),
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


def target_alignment_rows(
    target_probs: torch.Tensor,
    native_probs: torch.Tensor,
    mapped_probs: torch.Tensor,
    next_ids: Sequence[int] | None = None,
) -> list[dict[str, float | int]]:
    """Compare native and mapped draft proposals against the exact verifier.

    The speculative-decoding quantity of interest is verifier/proposal overlap, not
    whether mapped state reconstructs the draft's native distribution. Positive
    ``delta_target_alignment`` means the translated history moves the draft proposal
    distribution closer to the verifier on the same causal frontier.
    """
    if (
        target_probs.shape != native_probs.shape
        or target_probs.shape != mapped_probs.shape
        or target_probs.ndim != 2
    ):
        raise ValueError("target/native/mapped probabilities must share [prefix,vocab] shape")
    if next_ids is not None and len(next_ids) != target_probs.shape[0]:
        raise ValueError("next_ids must contain one token per probability row")

    target = _normalize_probabilities(target_probs)
    native = _normalize_probabilities(native_probs)
    mapped = _normalize_probabilities(mapped_probs)
    rows: list[dict[str, float | int]] = []
    for index, (p, q_native, q_mapped) in enumerate(
        zip(target, native, mapped, strict=True)
    ):
        native_overlap = _overlap(p, q_native)
        mapped_overlap = _overlap(p, q_mapped)
        row: dict[str, float | int] = {
            "a_target_native": float(native_overlap),
            "a_target_mapped": float(mapped_overlap),
            "delta_target_alignment": float(mapped_overlap - native_overlap),
            "kl_target_native": float(_kl(p, q_native)),
            "kl_target_mapped": float(_kl(p, q_mapped)),
            "top1_target_native": int(p.argmax() == q_native.argmax()),
            "top1_target_mapped": int(p.argmax() == q_mapped.argmax()),
        }
        if next_ids is not None:
            token = int(next_ids[index])
            if not 0 <= token < p.numel():
                raise ValueError("next token ID lies outside the probability vocabulary")
            row.update(
                {
                    "target_next_token_nll": float(-p[token].clamp_min(1e-12).log()),
                    "native_next_token_nll": float(-q_native[token].clamp_min(1e-12).log()),
                    "mapped_next_token_nll": float(-q_mapped[token].clamp_min(1e-12).log()),
                }
            )
        rows.append(row)
    return rows


def classify_transfer_gate(ci_low: float, ci_high: float, threshold: float = 0.95) -> str:
    """Apply the preregistered confidence-interval native-fidelity decision."""
    if not 0.0 <= ci_low <= ci_high <= 1.0:
        raise ValueError("transfer confidence interval must lie inside [0, 1]")
    if not 0.0 < threshold < 1.0:
        raise ValueError("transfer threshold must lie strictly inside (0, 1)")
    if ci_low > threshold:
        return "pass"
    if ci_high <= threshold:
        return "fail"
    return "inconclusive"


def classify_target_alignment_gate(ci_low: float, ci_high: float) -> str:
    """Classify whether mapping significantly helps or harms verifier alignment."""
    if not torch.isfinite(torch.tensor([ci_low, ci_high])).all() or ci_low > ci_high:
        raise ValueError("target-alignment confidence interval must be finite and ordered")
    if ci_low > 0.0:
        return "support"
    if ci_high < 0.0:
        return "harm"
    return "inconclusive"


def target_alignment_decision(target_status: str, fidelity_status: str) -> str:
    """Return the experiment action; native fidelity never vetoes target support."""
    if fidelity_status not in {"pass", "fail", "inconclusive", "incomplete"}:
        raise ValueError("unsupported native-fidelity status")
    if target_status == "support":
        return "go_sd"
    if target_status == "harm":
        return "stop_pair"
    if target_status == "inconclusive":
        return "expand"
    if target_status == "incomplete":
        return "incomplete"
    raise ValueError("unsupported target-alignment status")


def _bootstrap_mean_interval(
    values: torch.Tensor,
    *,
    samples: int,
    seed: int,
    cluster_ids: Sequence[int | str] | None = None,
    confidence: float = 0.95,
) -> tuple[float, float]:
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    if cluster_ids is None:
        cluster_ids = list(range(values.numel()))
    if len(cluster_ids) != values.numel():
        raise ValueError("cluster_ids must contain one ID per row")
    ordered = list(dict.fromkeys(cluster_ids))
    if not ordered:
        raise ValueError("at least one bootstrap cluster is required")
    cluster_index = {cluster: index for index, cluster in enumerate(ordered)}
    sums = torch.zeros(len(ordered), dtype=torch.float64)
    counts = torch.zeros(len(ordered), dtype=torch.float64)
    for value, cluster in zip(values, cluster_ids, strict=True):
        index = cluster_index[cluster]
        sums[index] += value
        counts[index] += 1
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randint(0, len(ordered), (samples, len(ordered)), generator=generator)
    means = sums[indices].sum(dim=1) / counts[indices].sum(dim=1)
    tail = (1.0 - confidence) / 2.0
    return float(torch.quantile(means, tail)), float(torch.quantile(means, 1.0 - tail))


def summarize_transfer(
    rows: Sequence[dict],
    *,
    requested: int,
    samples: int = 10000,
    seed: int = 0,
    threshold: float = 0.95,
    cluster_ids: Sequence[int | str] | None = None,
) -> dict:
    """Aggregate native-fidelity rows and withhold a gate from incomplete runs."""
    if requested <= 0:
        raise ValueError("requested row count must be positive")
    if not rows:
        raise ValueError("at least one transfer row is required")
    values = torch.tensor([float(row["a_transfer"]) for row in rows], dtype=torch.float64)
    if not torch.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("a_transfer rows must be finite and lie inside [0, 1]")
    if cluster_ids is None:
        cluster_ids = list(range(len(rows)))
    ci_low, ci_high = _bootstrap_mean_interval(
        values, samples=samples, seed=seed, cluster_ids=cluster_ids
    )
    complete = len(rows) == requested
    status = classify_transfer_gate(ci_low, ci_high, threshold) if complete else "incomplete"
    summary = {
        "mean_a_transfer": float(values.mean()),
        "median_a_transfer": float(values.median()),
        "p05_a_transfer": float(torch.quantile(values, 0.05)),
        "min_a_transfer": float(values.min()),
        "mean_kl_native_mapped": sum(float(row["kl_native_mapped"]) for row in rows) / len(rows),
        "top1_agreement": sum(int(row["top1_agreement"]) for row in rows) / len(rows),
    }
    if all("next_token_nll_delta" in row for row in rows):
        summary["mean_next_token_nll_delta"] = sum(
            float(row["next_token_nll_delta"]) for row in rows
        ) / len(rows)
    return {
        "requested_rows": requested,
        "completed_rows": len(rows),
        "independent_clusters": len(set(cluster_ids)),
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


def summarize_target_alignment(
    rows: Sequence[dict],
    *,
    requested: int,
    samples: int = 10000,
    seed: int = 0,
    cluster_ids: Sequence[int | str] | None = None,
) -> dict:
    """Aggregate paired verifier-alignment gains by source-document cluster."""
    if requested <= 0:
        raise ValueError("requested row count must be positive")
    if not rows:
        raise ValueError("at least one target-alignment row is required")
    deltas = torch.tensor([float(row["delta_target_alignment"]) for row in rows], dtype=torch.float64)
    native = torch.tensor([float(row["a_target_native"]) for row in rows], dtype=torch.float64)
    mapped = torch.tensor([float(row["a_target_mapped"]) for row in rows], dtype=torch.float64)
    if not torch.isfinite(deltas).all():
        raise ValueError("target-alignment deltas must be finite")
    if (
        not torch.isfinite(native).all()
        or not torch.isfinite(mapped).all()
        or (native < 0).any()
        or (native > 1).any()
        or (mapped < 0).any()
        or (mapped > 1).any()
    ):
        raise ValueError("target overlaps must be finite and lie inside [0, 1]")
    if cluster_ids is None:
        cluster_ids = list(range(len(rows)))
    ci_low, ci_high = _bootstrap_mean_interval(
        deltas, samples=samples, seed=seed, cluster_ids=cluster_ids
    )
    complete = len(rows) == requested
    status = classify_target_alignment_gate(ci_low, ci_high) if complete else "incomplete"
    summary = {
        "mean_a_target_native": float(native.mean()),
        "mean_a_target_mapped": float(mapped.mean()),
        "mean_delta_target_alignment": float(deltas.mean()),
        "median_delta_target_alignment": float(deltas.median()),
        "p05_delta_target_alignment": float(torch.quantile(deltas, 0.05)),
        "mapped_over_native_target_overlap": float(mapped.mean() / native.mean().clamp_min(1e-12)),
    }
    optional_means = {
        "kl_target_native": "mean_kl_target_native",
        "kl_target_mapped": "mean_kl_target_mapped",
        "top1_target_native": "top1_target_native",
        "top1_target_mapped": "top1_target_mapped",
        "target_next_token_nll": "mean_target_next_token_nll",
        "native_next_token_nll": "mean_native_next_token_nll",
        "mapped_next_token_nll": "mean_mapped_next_token_nll",
    }
    for key, output_key in optional_means.items():
        if all(key in row for row in rows):
            summary[output_key] = sum(float(row[key]) for row in rows) / len(rows)
    return {
        "requested_rows": requested,
        "completed_rows": len(rows),
        "independent_clusters": len(set(cluster_ids)),
        "summary": summary,
        "gate": {
            "status": status,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "confidence": 0.95,
            "bootstrap_samples": samples,
        },
    }


def finalize_screen(
    rows: Sequence[dict],
    *,
    requested: int,
    samples: int = 10000,
    seed: int = 0,
    threshold: float = 0.95,
    cluster_ids: Sequence[int | str] | None = None,
) -> dict:
    """Backward-compatible native-fidelity screen summary."""
    return summarize_transfer(
        rows,
        requested=requested,
        samples=samples,
        seed=seed,
        threshold=threshold,
        cluster_ids=cluster_ids,
    )


def validate_screen_inputs(mapper_metadata: dict, screen_manifest: dict) -> None:
    """Require a held-out verifier capture compatible with target-alignment screening."""
    if mapper_metadata.get("pair") != screen_manifest.get("pair"):
        raise ValueError("mapper and screen describe different model pairs")
    source = mapper_metadata.get("source_model", {})
    captured = screen_manifest.get("model", {})
    if source != captured:
        raise ValueError("screen verifier contract differs from mapper calibration")
    if mapper_metadata.get("dtype") != screen_manifest.get("dtype"):
        raise ValueError("screen dtype differs from mapper calibration")
    distribution = screen_manifest.get("target_distribution")
    if not isinstance(distribution, dict) or distribution.get("storage_dtype") != "float32" or float(
        distribution.get("temperature", float("nan"))
    ) != 1.0:
        raise ValueError(
            "screen target distribution contract must use FP32 storage at temperature 1.0"
        )
    validate_no_row_overlap(
        mapper_metadata.get("token_row_digests", []),
        screen_manifest.get("token_row_digests", []),
    )


def attention_output_cosines(
    native_outputs: Sequence[torch.Tensor], mapped_outputs: Sequence[torch.Tensor]
) -> list[float]:
    """Return flattened cosine similarity for matching draft attention layers."""
    if len(native_outputs) != len(mapped_outputs) or not native_outputs:
        raise ValueError("attention outputs must be non-empty and layer-aligned")
    result = []
    for native, mapped in zip(native_outputs, mapped_outputs, strict=True):
        if native.shape != mapped.shape:
            raise ValueError("matching attention outputs must have identical shapes")
        left, right = native.float().flatten(), mapped.float().flatten()
        if left.norm() == 0 or right.norm() == 0:
            raise ValueError("attention output cosine is undefined for a zero vector")
        result.append(float(torch.nn.functional.cosine_similarity(left, right, dim=0)))
    return result
