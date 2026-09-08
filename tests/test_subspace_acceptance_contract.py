import pytest

from verifier_anchored_sd.subspace_acceptance import validate_subspace_winner


def _result():
    return {
        "decision": {"status": "go_e2"},
        "pair": {"target": "T", "draft": "D"},
        "protocol": {
            "mapper_checkpoint_sha256": "mapper-sha",
            "subspace_artifact_sha256": "basis-sha",
            "evaluation_token_row_digests": ["eval-a", "eval-b"],
        },
        "subspace_metadata": {
            "fit_token_row_digests": ["fit-a", "fit-b"],
        },
        "winner": {
            "method": "benefit_positive_mapped_soft_r16_b0.25",
            "deployment_valid": True,
            "rank": 16,
            "spec": {
                "mode": "mapped_soft",
                "family": "benefit_positive",
                "rank": 16,
                "beta": 0.25,
                "alpha": None,
            },
        },
    }


def test_winner_contract_returns_frozen_deployment_spec_and_forbidden_rows():
    contract = validate_subspace_winner(
        _result(), mapper_sha256="mapper-sha", subspace_sha256="basis-sha"
    )

    assert contract["spec"].family == "benefit_positive"
    assert contract["spec"].rank == 16
    assert contract["spec"].beta == 0.25
    assert contract["forbidden_token_row_digests"] == {"fit-a", "fit-b", "eval-a", "eval-b"}


def test_winner_contract_rejects_non_go_result():
    result = _result()
    result["decision"]["status"] = "no_deployment_winner"

    with pytest.raises(RuntimeError, match="go_e2"):
        validate_subspace_winner(result, mapper_sha256="mapper-sha", subspace_sha256="basis-sha")


def test_winner_contract_rejects_delta_upper_bound():
    result = _result()
    result["winner"]["spec"] = {
        "mode": "delta_projected",
        "family": "benefit_positive",
        "rank": 16,
        "alpha": 1.0,
        "beta": None,
    }

    with pytest.raises(RuntimeError, match="mapped-only"):
        validate_subspace_winner(result, mapper_sha256="mapper-sha", subspace_sha256="basis-sha")


def test_winner_contract_rejects_mapper_or_basis_digest_mismatch():
    with pytest.raises(RuntimeError, match="mapper"):
        validate_subspace_winner(_result(), mapper_sha256="other", subspace_sha256="basis-sha")
    with pytest.raises(RuntimeError, match="subspace"):
        validate_subspace_winner(_result(), mapper_sha256="mapper-sha", subspace_sha256="other")


def test_winner_contract_rejects_missing_eval_row_provenance():
    result = _result()
    result["protocol"].pop("evaluation_token_row_digests")

    with pytest.raises(RuntimeError, match="row provenance"):
        validate_subspace_winner(result, mapper_sha256="mapper-sha", subspace_sha256="basis-sha")
