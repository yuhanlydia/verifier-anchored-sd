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


def test_confirmatory_eval_requires_a_passing_distribution_screen():
    with pytest.raises(RuntimeError, match="did not pass"):
        validate_screen_gate({"gate": {"status": "fail"}})

    validate_screen_gate({"gate": {"status": "pass"}})
    validate_screen_gate({"gate": {"status": "fail"}}, allow_failed=True)
