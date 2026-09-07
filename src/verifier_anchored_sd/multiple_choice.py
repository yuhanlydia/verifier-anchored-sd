"""Multiple-choice scoring helpers for mapped-cache confirmation checks."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def validate_screen_gate(screen_result: dict, *, allow_failed: bool = False) -> None:
    """Require target-alignment support before confirmatory task evaluation.

    Native-draft fidelity is diagnostic only.  A mapped state may differ from the
    native draft yet still be useful for speculative decoding when it moves the
    proposal distribution toward the verifier.  Diagnostic overrides remain
    available explicitly through ``allow_failed``.
    """
    if allow_failed:
        return
    target_status = screen_result.get("target_alignment", {}).get("gate", {}).get("status")
    decision = screen_result.get("decision", {}).get("status")
    if target_status != "support" or decision != "go_sd":
        raise RuntimeError(
            "target alignment did not support confirmatory evaluation "
            f"(target_status={target_status!r}, decision={decision!r}); "
            "use --allow-failed-screen only for a diagnostic run"
        )
    requested = screen_result.get("requested_rows")
    completed = screen_result.get("completed_rows")
    if not isinstance(requested, int) or requested <= 0 or completed != requested:
        raise RuntimeError("target-alignment screen must be complete before confirmation")


def choice_nll(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    context_length: int,
) -> float:
    """Return mean causal NLL over continuation tokens only.

    ``logits`` contains predictions after each input position. ``labels`` is the
    full context-plus-continuation sequence, so continuation label ``i`` is
    scored from logit position ``i - 1``.
    """
    if logits.ndim == 3:
        if logits.shape[0] != 1:
            raise ValueError("batched choice scoring requires batch size one")
        logits = logits[0]
    if logits.ndim != 2 or labels.ndim != 1:
        raise ValueError("logits must be [steps,vocab] and labels must be [tokens]")
    if not 0 < context_length < labels.numel():
        raise ValueError("context_length must leave at least one continuation token")
    last_prediction = labels.numel() - 1
    if logits.shape[0] < last_prediction:
        raise ValueError("logits do not cover every continuation token")

    prediction_logits = logits[context_length - 1 : last_prediction].float()
    continuation = labels[context_length:].to(prediction_logits.device)
    return float(F.cross_entropy(prediction_logits, continuation, reduction="mean"))


def normalized_retention(
    native_accuracy: float,
    mapped_accuracy: float,
    *,
    random_floor: float,
) -> float:
    """Return floor-normalized mapped/native multiple-choice retention."""
    values = (native_accuracy, mapped_accuracy, random_floor)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("accuracies and random floor must be finite")
    if not 0.0 <= native_accuracy <= 1.0 or not 0.0 <= mapped_accuracy <= 1.0:
        raise ValueError("accuracies must lie in [0, 1]")
    if not 0.0 <= random_floor < 1.0:
        raise ValueError("random floor must lie in [0, 1)")
    if native_accuracy <= random_floor:
        raise ValueError("native accuracy must be above random floor")
    return (mapped_accuracy - random_floor) / (native_accuracy - random_floor)
