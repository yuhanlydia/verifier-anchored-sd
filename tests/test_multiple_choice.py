import pytest
import torch

from verifier_anchored_sd.multiple_choice import (
    choice_nll,
    normalized_retention,
    validate_screen_gate,
)


def test_choice_nll_scores_only_continuation_tokens():
    logits = torch.zeros(2, 6)
    logits[0, 4] = 20
    logits[1, 5] = 20
    labels = torch.tensor([3, 4, 5])

    assert choice_nll(logits, labels, context_length=1) < 1e-6


def test_choice_nll_averages_instead_of_favoring_short_choices():
    short_logits = torch.tensor([[0.0, 2.0]])
    long_logits = torch.tensor([[0.0, 2.0], [0.0, 2.0]])

    short = choice_nll(short_logits, torch.tensor([0, 1]), context_length=1)
    long = choice_nll(long_logits, torch.tensor([0, 1, 1]), context_length=1)

    assert short == pytest.approx(long)


def test_floor_normalized_retention():
    assert normalized_retention(0.75, 0.70, random_floor=0.25) == pytest.approx(0.9)


def test_floor_normalized_retention_requires_native_above_floor():
    with pytest.raises(ValueError, match="above random floor"):
        normalized_retention(0.25, 0.25, random_floor=0.25)


def _supported_screen(native_fidelity: str = "fail"):
    return {
        "native_fidelity": {"gate": {"status": native_fidelity}},
        "target_alignment": {"gate": {"status": "support"}},
        "decision": {"status": "go_sd"},
        "requested_rows": 128,
        "completed_rows": 128,
    }


def test_confirmatory_eval_uses_target_alignment_not_native_fidelity():
    # Native reconstruction may fail while mapping still moves the draft toward the verifier.
    validate_screen_gate(_supported_screen(native_fidelity="fail"))

    harm = _supported_screen()
    harm["target_alignment"]["gate"]["status"] = "harm"
    harm["decision"]["status"] = "stop_pair"
    with pytest.raises(RuntimeError, match="target alignment"):
        validate_screen_gate(harm)

    legacy = {"gate": {"status": "pass"}, "requested_rows": 128, "completed_rows": 128}
    with pytest.raises(RuntimeError, match="target alignment"):
        validate_screen_gate(legacy)


def test_confirmatory_eval_requires_complete_target_alignment_screen():
    incomplete = _supported_screen()
    incomplete["completed_rows"] = 17
    with pytest.raises(RuntimeError, match="complete"):
        validate_screen_gate(incomplete)


def test_diagnostic_override_allows_failed_or_legacy_screen():
    validate_screen_gate({"gate": {"status": "fail"}}, allow_failed=True)
