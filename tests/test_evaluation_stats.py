import torch

from verifier_anchored_sd.evaluation import (
    expected_accepted_length,
    paired_bootstrap_mean_difference,
)


def test_expected_accepted_length_matches_prefix_acceptance_mass():
    p = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    q = torch.tensor([[1.0, 0.0], [0.5, 0.5]])
    # alpha=[0.5,1.0], so P(accept >=1)=0.5 and P(accept >=2)=0.5.
    assert torch.allclose(expected_accepted_length(p, q), torch.tensor(1.0))


def test_paired_bootstrap_reports_positive_mean_difference():
    a = [2.0, 3.0, 4.0, 5.0]
    b = [1.0, 2.0, 3.0, 4.0]
    result = paired_bootstrap_mean_difference(a, b, samples=1000, seed=7)
    assert result["mean_difference"] == 1.0
    assert result["ci_low"] > 0.0
    assert result["ci_high"] > 0.0
