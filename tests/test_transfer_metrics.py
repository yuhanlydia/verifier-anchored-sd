import pytest
import torch

from verifier_anchored_sd.transfer_metrics import (
    classify_transfer_gate,
    distribution_transfer_rows,
    summarize_transfer,
)


def test_identical_distributions_have_perfect_transfer():
    probs = torch.tensor([[0.25, 0.75], [0.5, 0.5]])

    rows = distribution_transfer_rows(probs, probs)

    assert [row["a_transfer"] for row in rows] == [1.0, 1.0]
    assert [row["kl_native_mapped"] for row in rows] == [0.0, 0.0]
    assert [row["top1_agreement"] for row in rows] == [1, 1]


def test_transfer_metrics_match_hand_computed_binary_case():
    native = torch.tensor([[0.75, 0.25]])
    mapped = torch.tensor([[0.25, 0.75]])

    row = distribution_transfer_rows(native, mapped, next_ids=[0])[0]

    assert row["a_transfer"] == pytest.approx(0.5)
    assert row["kl_native_mapped"] == pytest.approx(0.5 * torch.log(torch.tensor(3.0)).item())
    assert row["top1_agreement"] == 0
    assert row["next_token_nll_delta"] == pytest.approx(torch.log(torch.tensor(3.0)).item())


def test_nonfinite_probabilities_are_rejected():
    with pytest.raises(ValueError, match="finite"):
        distribution_transfer_rows(
            torch.tensor([[float("nan"), 1.0]]),
            torch.tensor([[0.5, 0.5]]),
        )


def test_non_normalized_probability_rows_are_rejected():
    with pytest.raises(ValueError, match="sum to one"):
        distribution_transfer_rows(
            torch.tensor([[0.25, 0.25]]),
            torch.tensor([[0.5, 0.5]]),
        )


def test_gate_distinguishes_pass_fail_and_inconclusive():
    assert classify_transfer_gate(0.951, 0.970, threshold=0.95) == "pass"
    assert classify_transfer_gate(0.920, 0.950, threshold=0.95) == "fail"
    assert classify_transfer_gate(0.940, 0.960, threshold=0.95) == "inconclusive"


def test_summary_reports_incomplete_before_scientific_gate():
    rows = [
        {"a_transfer": 0.99, "kl_native_mapped": 0.01, "top1_agreement": 1},
    ]

    result = summarize_transfer(rows, requested=2, samples=100, seed=0, threshold=0.95)

    assert result["gate"]["status"] == "incomplete"
    assert result["completed_rows"] == 1


def test_summary_cluster_bootstrap_reports_independent_documents():
    rows = [
        {"a_transfer": 0.2, "kl_native_mapped": 1.0, "top1_agreement": 0},
        {"a_transfer": 0.4, "kl_native_mapped": 1.0, "top1_agreement": 0},
        {"a_transfer": 0.8, "kl_native_mapped": 0.1, "top1_agreement": 1},
    ]

    result = summarize_transfer(
        rows,
        requested=3,
        samples=100,
        seed=0,
        cluster_ids=[0, 0, 1],
    )

    assert result["independent_clusters"] == 2


def test_one_cluster_bootstrap_collapses_to_observed_mean():
    rows = [
        {"a_transfer": 0.2, "kl_native_mapped": 1.0, "top1_agreement": 0},
        {"a_transfer": 0.4, "kl_native_mapped": 1.0, "top1_agreement": 0},
    ]

    result = summarize_transfer(
        rows,
        requested=2,
        samples=100,
        seed=0,
        cluster_ids=[7, 7],
    )

    assert result["gate"]["ci_low"] == pytest.approx(0.3)
    assert result["gate"]["ci_high"] == pytest.approx(0.3)
