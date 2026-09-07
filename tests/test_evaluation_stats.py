import pytest
import torch

from verifier_anchored_sd.evaluation import (
    expected_accepted_length,
    paired_bootstrap_mean_difference,
)


def test_expected_accepted_length_matches_prefix_acceptance_mass():
    p = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    q = torch.tensor([[1.0, 0.0], [0.5, 0.5]])
    # The realized first proposal is token 0, whose acceptance probability is 0.5.
    assert torch.allclose(expected_accepted_length(p, q, [0, 0]), torch.tensor(1.0))


def test_expected_accepted_length_conditions_on_realized_proposals():
    p = torch.tensor([[0.9, 0.1]])
    q = torch.tensor([[0.5, 0.5]])

    # Distribution overlap is 0.6, but proposal token 0 is accepted with probability 1.
    assert torch.allclose(expected_accepted_length(p, q, [0]), torch.tensor(1.0))


def test_expected_accepted_length_supports_batched_proposals():
    p = torch.tensor([[[0.9, 0.1]], [[0.1, 0.9]]])
    q = torch.tensor([[[0.5, 0.5]], [[0.5, 0.5]]])

    result = expected_accepted_length(p, q, torch.tensor([[0], [0]]))

    assert torch.allclose(result, torch.tensor([1.0, 0.2]))


def test_paired_bootstrap_reports_positive_mean_difference():
    a = [2.0, 3.0, 4.0, 5.0]
    b = [1.0, 2.0, 3.0, 4.0]
    result = paired_bootstrap_mean_difference(a, b, samples=1000, seed=7)
    assert result["mean_difference"] == 1.0
    assert result["ci_low"] > 0.0
    assert result["ci_high"] > 0.0


def test_paired_bootstrap_resamples_document_clusters():
    result = paired_bootstrap_mean_difference(
        [0.2, 0.4],
        [0.1, 0.1],
        samples=100,
        seed=0,
        cluster_ids=[5, 5],
    )

    assert result["ci_low"] == pytest.approx(0.2)
    assert result["ci_high"] == pytest.approx(0.2)
