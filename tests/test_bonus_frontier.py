import torch

from verifier_anchored_sd.spec_decode.exact_sd import SpeculativeResult, choose_frontier_token


def test_rejection_uses_exact_correction_as_frontier():
    result = SpeculativeResult([1], correction=3, rejected_at=1)
    token, kind = choose_frontier_token(result, torch.tensor([0.0, 1.0, 0.0, 0.0]))
    assert token == 3
    assert kind == "correction"


def test_all_accepted_samples_target_bonus_frontier():
    result = SpeculativeResult([1, 2, 3], correction=None, rejected_at=None)
    token, kind = choose_frontier_token(result, torch.tensor([0.0, 0.0, 1.0, 0.0]))
    assert token == 2
    assert kind == "bonus"
