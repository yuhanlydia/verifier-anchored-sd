"""Provenance and decision gates from subspace screening to block-level SD."""

from __future__ import annotations

import math

from .experiment_artifacts import validate_no_row_overlap
from .kv_subspace import InterventionSpec


def acceptance_method_matrix() -> dict[str, dict[str, str]]:
    """Return the frozen five-policy block-level comparison in stable order."""
    return {
        "native_sd": {
            "mapper": "base",
            "init_mode": "native",
            "refresh_policy": "none",
        },
        "full_mapped_init_only": {
            "mapper": "base",
            "init_mode": "mapped_native_frontier",
            "refresh_policy": "none",
        },
        "full_mapped_accepted_only": {
            "mapper": "base",
            "init_mode": "mapped_native_frontier",
            "refresh_policy": "accepted_only",
        },
        "subspace_mapped_init_only": {
            "mapper": "subspace",
            "init_mode": "mapped_native_frontier",
            "refresh_policy": "none",
        },
        "subspace_mapped_accepted_only": {
            "mapper": "subspace",
            "init_mode": "mapped_native_frontier",
            "refresh_policy": "accepted_only",
        },
    }


def classify_block_delta(ci_low: float, ci_high: float) -> str:
    """Classify a paired block-level method-minus-baseline confidence interval."""
    if not math.isfinite(ci_low) or not math.isfinite(ci_high) or ci_low > ci_high:
        raise ValueError("block-level confidence interval must be finite and ordered")
    if ci_low > 0:
        return "support"
    if ci_high < 0:
        return "stop"
    return "inconclusive"


def _nonempty_digest_set(value, *, name: str) -> set[str]:
    if not isinstance(value, list) or not value:
        raise RuntimeError(f"subspace winner lacks {name} row provenance")
    result = {str(item) for item in value if str(item)}
    if len(result) != len(value):
        raise RuntimeError(f"subspace winner has invalid or duplicate {name} row provenance")
    return result


def validate_subspace_winner(
    result: dict,
    *,
    mapper_sha256: str,
    subspace_sha256: str,
) -> dict:
    """Return one frozen mapped-only winner and all B/C rows forbidden to E2.

    The intervention sweep itself selects the configuration. This function does not
    re-rank candidates; it only verifies that the selected method was eligible, is
    reproducibly bound to the supplied mapper/basis artifacts, and does not require
    native draft history.
    """
    if result.get("decision", {}).get("status") != "go_e2":
        raise RuntimeError("subspace intervention result must have decision.status='go_e2'")
    winner = result.get("winner")
    if not isinstance(winner, dict) or not winner.get("deployment_valid", False):
        raise RuntimeError("subspace intervention result lacks a deployment-valid winner")

    protocol = result.get("protocol")
    if not isinstance(protocol, dict):
        raise RuntimeError("subspace intervention result lacks protocol provenance")
    if protocol.get("mapper_checkpoint_sha256") != mapper_sha256:
        raise RuntimeError("subspace winner mapper checkpoint differs from supplied mapper")
    if protocol.get("subspace_artifact_sha256") != subspace_sha256:
        raise RuntimeError("subspace winner subspace artifact differs from supplied basis")

    spec_value = winner.get("spec")
    if not isinstance(spec_value, dict):
        raise RuntimeError("subspace winner lacks a frozen intervention spec")
    try:
        spec = InterventionSpec(
            mode=spec_value["mode"],
            family=str(spec_value["family"]),
            rank=int(spec_value["rank"]),
            beta=spec_value.get("beta"),
            alpha=spec_value.get("alpha"),
        )
        spec.validate()
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("subspace winner contains an invalid intervention spec") from exc
    if spec.mode not in {"mapped_soft", "orthogonal"}:
        raise RuntimeError(
            "subspace E2 requires a mapped-only deployment intervention; "
            f"winner mode {spec.mode!r} needs native draft history"
        )

    subspace_metadata = result.get("subspace_metadata")
    if not isinstance(subspace_metadata, dict):
        raise RuntimeError("subspace winner lacks basis-fit provenance")
    fit_rows = _nonempty_digest_set(
        subspace_metadata.get("fit_token_row_digests"), name="basis-fit"
    )
    eval_rows = _nonempty_digest_set(
        protocol.get("evaluation_token_row_digests"), name="intervention-evaluation"
    )
    validate_no_row_overlap(fit_rows, eval_rows)

    pair = result.get("pair")
    if not isinstance(pair, dict) or not pair.get("target") or not pair.get("draft"):
        raise RuntimeError("subspace winner lacks model-pair provenance")

    return {
        "spec": spec,
        "method": str(winner.get("method", "")),
        "pair": dict(pair),
        "forbidden_token_row_digests": fit_rows | eval_rows,
    }
