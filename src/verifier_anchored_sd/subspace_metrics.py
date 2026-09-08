"""Statistics and model-selection rules for KV subspace intervention sweeps."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from .evaluation import paired_bootstrap_mean_difference


def _method_rows(rows: Sequence[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        method = row.get("method")
        if not isinstance(method, str) or not method:
            raise ValueError("every subspace evaluation row needs a non-empty method")
        grouped[method].append(dict(row))
    return dict(grouped)


def summarize_methods(rows: Sequence[dict], *, expected_prompts: int) -> dict[str, dict]:
    """Aggregate one complete row per prompt for every method."""
    if expected_prompts <= 0:
        raise ValueError("expected_prompts must be positive")
    grouped = _method_rows(rows)
    result: dict[str, dict] = {}
    for method, subset in grouped.items():
        prompts = [int(row["prompt"]) for row in subset]
        if len(subset) != expected_prompts or len(set(prompts)) != expected_prompts:
            raise RuntimeError(
                f"method {method!r} has {len(set(prompts))}/{expected_prompts} complete prompts"
            )
        for field in ("a_target", "kl_target", "top1_target", "next_token_nll"):
            if not all(field in row for row in subset):
                raise RuntimeError(f"method {method!r} is missing metric {field}")
        result[method] = {
            "prompts": expected_prompts,
            "mean_a_target": sum(float(row["a_target"]) for row in subset) / expected_prompts,
            "mean_kl_target": sum(float(row["kl_target"]) for row in subset) / expected_prompts,
            "target_top1_agreement": sum(int(row["top1_target"]) for row in subset)
            / expected_prompts,
            "mean_next_token_nll": sum(float(row["next_token_nll"]) for row in subset)
            / expected_prompts,
        }
        if all("elapsed_s" in row for row in subset):
            result[method]["mean_elapsed_s"] = sum(float(row["elapsed_s"]) for row in subset) / expected_prompts
    return result


def paired_method_difference(
    rows: Sequence[dict],
    *,
    method: str,
    baseline: str,
    metric: str,
    expected_prompts: int,
    samples: int = 10000,
    seed: int = 0,
) -> dict:
    """Paired document-cluster bootstrap for method minus baseline on one metric."""
    by_method: dict[int, dict] = {}
    by_baseline: dict[int, dict] = {}
    for row in rows:
        name = row.get("method")
        prompt = int(row["prompt"])
        if name == method:
            by_method[prompt] = row
        elif name == baseline:
            by_baseline[prompt] = row
    prompts = sorted(set(by_method) & set(by_baseline))
    if len(prompts) != expected_prompts:
        raise RuntimeError(
            f"paired method metric has {len(prompts)}/{expected_prompts} complete prompts"
        )
    method_values = [float(by_method[prompt][metric]) for prompt in prompts]
    baseline_values = [float(by_baseline[prompt][metric]) for prompt in prompts]
    cluster_ids = []
    for prompt in prompts:
        left = by_method[prompt].get("document_id")
        right = by_baseline[prompt].get("document_id")
        if left is None or right is None or left != right:
            raise RuntimeError("paired rows must share one document_id per prompt")
        cluster_ids.append(left)
    result = paired_bootstrap_mean_difference(
        method_values,
        baseline_values,
        samples=samples,
        seed=seed,
        cluster_ids=cluster_ids,
    )
    result.update({"method": method, "baseline": baseline, "metric": metric})
    return result


def select_deployment_winner(candidates: dict[str, dict]) -> dict | None:
    """Select one mapped-only candidate after the preregistered dual-baseline gate.

    A candidate is eligible only if it is deployment-valid and its target-overlap
    paired CI lower bound is strictly positive versus both pure native 4B and the
    unfiltered full-mapped cache. Among eligible candidates: maximize mean target
    overlap, then minimize target KL, then minimize rank, then method name for a
    deterministic final tie break.
    """
    eligible = []
    for key, candidate in candidates.items():
        if not candidate.get("deployment_valid", False):
            continue
        vs_native = candidate.get("vs_native", {})
        vs_full = candidate.get("vs_full_mapped", {})
        if float(vs_native.get("ci_low", float("-inf"))) <= 0:
            continue
        if float(vs_full.get("ci_low", float("-inf"))) <= 0:
            continue
        if "mean_a_target" not in candidate or "mean_kl_target" not in candidate:
            raise ValueError(f"eligible candidate {key!r} lacks primary summary metrics")
        if "rank" not in candidate:
            raise ValueError(f"eligible candidate {key!r} lacks rank")
        eligible.append(dict(candidate))
    if not eligible:
        return None
    eligible.sort(
        key=lambda candidate: (
            -float(candidate["mean_a_target"]),
            float(candidate["mean_kl_target"]),
            int(candidate["rank"]),
            str(candidate.get("method", "")),
        )
    )
    return eligible[0]
