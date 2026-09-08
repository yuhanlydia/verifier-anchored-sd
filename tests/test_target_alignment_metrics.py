import pytest
import torch

from verifier_anchored_sd.transfer_metrics import (
    summarize_target_alignment,
    target_alignment_decision,
    target_alignment_rows,
)


def test_mapped_distribution_closer_to_target_has_positive_delta():
    target = torch.tensor([[0.9, 0.1]])
    native = torch.tensor([[0.6, 0.4]])
    mapped = torch.tensor([[0.8, 0.2]])

    row = target_alignment_rows(target, native, mapped, next_ids=[0])[0]

    assert row["a_target_native"] == pytest.approx(0.7)
    assert row["a_target_mapped"] == pytest.approx(0.9)
    assert row["delta_target_alignment"] == pytest.approx(0.2)
    assert row["top1_target_native"] == 1
    assert row["top1_target_mapped"] == 1
    assert row["target_next_token_nll"] < row["native_next_token_nll"]
    assert row["mapped_next_token_nll"] < row["native_next_token_nll"]


def test_target_alignment_summary_support_harm_and_inconclusive():
    support_rows = [
        {"delta_target_alignment": 0.10, "a_target_native": 0.6, "a_target_mapped": 0.7},
        {"delta_target_alignment": 0.20, "a_target_native": 0.5, "a_target_mapped": 0.7},
    ]
    harm_rows = [
        {"delta_target_alignment": -0.10, "a_target_native": 0.7, "a_target_mapped": 0.6},
        {"delta_target_alignment": -0.20, "a_target_native": 0.8, "a_target_mapped": 0.6},
    ]
    mixed_rows = [
        {"delta_target_alignment": -0.10, "a_target_native": 0.7, "a_target_mapped": 0.6},
        {"delta_target_alignment": 0.10, "a_target_native": 0.6, "a_target_mapped": 0.7},
    ]

    support = summarize_target_alignment(
        support_rows, requested=2, samples=200, seed=0, cluster_ids=[0, 1]
    )
    harm = summarize_target_alignment(
        harm_rows, requested=2, samples=200, seed=0, cluster_ids=[0, 1]
    )
    mixed = summarize_target_alignment(
        mixed_rows, requested=2, samples=200, seed=0, cluster_ids=[0, 1]
    )

    assert support["gate"]["status"] == "support"
    assert harm["gate"]["status"] == "harm"
    assert mixed["gate"]["status"] == "inconclusive"


def test_incomplete_target_alignment_withholds_decision():
    result = summarize_target_alignment(
        [{"delta_target_alignment": 0.5, "a_target_native": 0.4, "a_target_mapped": 0.9}],
        requested=2,
        samples=100,
        seed=0,
        cluster_ids=[0],
    )

    assert result["gate"]["status"] == "incomplete"


def test_target_support_allows_sd_even_when_native_fidelity_failed():
    assert target_alignment_decision("support", "fail") == "go_sd"
    assert target_alignment_decision("harm", "pass") == "stop_pair"
    assert target_alignment_decision("inconclusive", "fail") == "expand"
    assert target_alignment_decision("incomplete", "fail") == "incomplete"
