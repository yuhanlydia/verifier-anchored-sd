import pytest

from verifier_anchored_sd.evaluation import (
    acceptance_methods,
    classify_mapper_retention,
    classify_refresh_delta,
)


def test_e2_method_matrix_separates_legacy_and_primary_contrasts():
    methods = acceptance_methods()

    assert methods["legacy_full_refresh"] == {
        "init_mode": "legacy_mapped",
        "refresh_policy": "full",
    }
    assert methods["mapped_accepted_only"] == {
        "init_mode": "mapped_native_frontier",
        "refresh_policy": "accepted_only",
    }
    assert list(methods) == [
        "native_sd",
        "legacy_mapped_init_only",
        "mapped_init_only",
        "legacy_full_refresh",
        "mapped_accepted_only",
    ]


@pytest.mark.parametrize(
    ("retention", "expected"),
    [(0.90, "confirmatory"), (0.899, "exploratory"), (0.85, "exploratory"), (0.849, "reject")],
)
def test_mapper_retention_uses_preregistered_tiers(retention, expected):
    assert classify_mapper_retention(retention) == expected


@pytest.mark.parametrize(
    ("low", "high", "expected"),
    [(0.01, 0.10, "support"), (-0.10, -0.01, "stop"), (-0.01, 0.02, "inconclusive")],
)
def test_refresh_decision_requires_confidence_interval_sign(low, high, expected):
    assert classify_refresh_delta(low, high) == expected
