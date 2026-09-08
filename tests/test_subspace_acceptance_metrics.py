import pytest

from verifier_anchored_sd.subspace_acceptance import (
    acceptance_method_matrix,
    classify_block_delta,
)


def test_acceptance_method_matrix_contains_native_full_and_subspace_policies():
    methods = acceptance_method_matrix()

    assert list(methods) == [
        "native_sd",
        "full_mapped_init_only",
        "full_mapped_accepted_only",
        "subspace_mapped_init_only",
        "subspace_mapped_accepted_only",
    ]
    assert methods["native_sd"] == {
        "mapper": "base",
        "init_mode": "native",
        "refresh_policy": "none",
    }
    assert methods["subspace_mapped_accepted_only"]["mapper"] == "subspace"
    assert methods["subspace_mapped_accepted_only"]["refresh_policy"] == "accepted_only"


def test_block_delta_gate_requires_strictly_positive_or_negative_interval():
    assert classify_block_delta(0.01, 0.05) == "support"
    assert classify_block_delta(-0.08, -0.01) == "stop"
    assert classify_block_delta(-0.01, 0.03) == "inconclusive"
    assert classify_block_delta(0.0, 0.02) == "inconclusive"
    assert classify_block_delta(-0.02, 0.0) == "inconclusive"


def test_block_delta_gate_rejects_invalid_interval():
    with pytest.raises(ValueError, match="finite and ordered"):
        classify_block_delta(0.2, 0.1)
