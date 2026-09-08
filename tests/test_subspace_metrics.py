import pytest

from verifier_anchored_sd.subspace_metrics import (
    paired_method_difference,
    select_deployment_winner,
    summarize_methods,
)


def _rows():
    rows = []
    values = {
        "native": [0.70, 0.72, 0.68, 0.71],
        "full_mapped": [0.69, 0.70, 0.67, 0.69],
        "grad_r8_b025": [0.75, 0.76, 0.74, 0.75],
        "benefit_r8_b025": [0.77, 0.78, 0.76, 0.77],
        "delta_upper": [0.90, 0.91, 0.89, 0.90],
    }
    kls = {
        "native": [0.40] * 4,
        "full_mapped": [0.48] * 4,
        "grad_r8_b025": [0.34] * 4,
        "benefit_r8_b025": [0.30] * 4,
        "delta_upper": [0.10] * 4,
    }
    for method, scores in values.items():
        for prompt, score in enumerate(scores):
            rows.append(
                {
                    "method": method,
                    "prompt": prompt,
                    "document_id": prompt // 2,
                    "a_target": score,
                    "kl_target": kls[method][prompt],
                    "top1_target": int(score > 0.73),
                    "next_token_nll": 1.0 - score,
                }
            )
    return rows


def test_paired_method_difference_uses_same_prompt_rows_and_document_clusters():
    result = paired_method_difference(
        _rows(),
        method="grad_r8_b025",
        baseline="native",
        metric="a_target",
        expected_prompts=4,
        samples=500,
        seed=0,
    )

    assert result["mean_difference"] == pytest.approx(0.0475)
    assert result["ci_low"] > 0
    assert result["independent_clusters"] == 2


def test_paired_method_difference_rejects_incomplete_pairing():
    rows = [row for row in _rows() if not (row["method"] == "grad_r8_b025" and row["prompt"] == 3)]

    with pytest.raises(RuntimeError, match="3/4"):
        paired_method_difference(
            rows,
            method="grad_r8_b025",
            baseline="native",
            metric="a_target",
            expected_prompts=4,
            samples=100,
            seed=0,
        )


def test_method_summary_reports_exact_prompt_count_and_primary_metrics():
    result = summarize_methods(_rows(), expected_prompts=4)

    assert result["benefit_r8_b025"]["prompts"] == 4
    assert result["benefit_r8_b025"]["mean_a_target"] == pytest.approx(0.77)
    assert result["benefit_r8_b025"]["mean_kl_target"] == pytest.approx(0.30)
    assert result["benefit_r8_b025"]["target_top1_agreement"] == 1.0


def test_winner_must_beat_both_native_and_full_mapped_with_positive_ci():
    candidates = {
        "grad": {
            "method": "grad_r8_b025",
            "deployment_valid": True,
            "rank": 8,
            "mean_a_target": 0.75,
            "mean_kl_target": 0.34,
            "vs_native": {"ci_low": 0.01},
            "vs_full_mapped": {"ci_low": 0.02},
        },
        "benefit": {
            "method": "benefit_r8_b025",
            "deployment_valid": True,
            "rank": 8,
            "mean_a_target": 0.77,
            "mean_kl_target": 0.30,
            "vs_native": {"ci_low": 0.03},
            "vs_full_mapped": {"ci_low": 0.04},
        },
    }

    winner = select_deployment_winner(candidates)

    assert winner["method"] == "benefit_r8_b025"


def test_delta_upper_bound_cannot_be_selected_even_if_score_is_best():
    candidates = {
        "deploy": {
            "method": "benefit_r16_b0",
            "deployment_valid": True,
            "rank": 16,
            "mean_a_target": 0.78,
            "mean_kl_target": 0.29,
            "vs_native": {"ci_low": 0.02},
            "vs_full_mapped": {"ci_low": 0.03},
        },
        "upper": {
            "method": "benefit_delta_r16_a1",
            "deployment_valid": False,
            "rank": 16,
            "mean_a_target": 0.92,
            "mean_kl_target": 0.08,
            "vs_native": {"ci_low": 0.15},
            "vs_full_mapped": {"ci_low": 0.18},
        },
    }

    winner = select_deployment_winner(candidates)

    assert winner["method"] == "benefit_r16_b0"


def test_candidate_failing_native_gate_is_not_eligible():
    candidates = {
        "only": {
            "method": "grad_r8_b0",
            "deployment_valid": True,
            "rank": 8,
            "mean_a_target": 0.74,
            "mean_kl_target": 0.33,
            "vs_native": {"ci_low": -0.001},
            "vs_full_mapped": {"ci_low": 0.02},
        }
    }

    assert select_deployment_winner(candidates) is None


def test_winner_tie_breaks_by_lower_kl_then_lower_rank():
    candidates = {
        "a": {
            "method": "a",
            "deployment_valid": True,
            "rank": 32,
            "mean_a_target": 0.80,
            "mean_kl_target": 0.30,
            "vs_native": {"ci_low": 0.01},
            "vs_full_mapped": {"ci_low": 0.01},
        },
        "b": {
            "method": "b",
            "deployment_valid": True,
            "rank": 64,
            "mean_a_target": 0.80,
            "mean_kl_target": 0.25,
            "vs_native": {"ci_low": 0.01},
            "vs_full_mapped": {"ci_low": 0.01},
        },
        "c": {
            "method": "c",
            "deployment_valid": True,
            "rank": 8,
            "mean_a_target": 0.80,
            "mean_kl_target": 0.25,
            "vs_native": {"ci_low": 0.01},
            "vs_full_mapped": {"ci_low": 0.01},
        },
    }

    assert select_deployment_winner(candidates)["method"] == "c"
