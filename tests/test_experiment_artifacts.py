import json

import pytest

from verifier_anchored_sd.experiment_artifacts import (
    atomic_write_json,
    token_rows_digest,
    validate_disjoint,
    validate_no_row_overlap,
    validate_protocol_contract,
)


def test_atomic_json_replaces_complete_file_without_temporary_residue(tmp_path):
    path = tmp_path / "result.json"
    path.write_text('{"old": true}\n')

    atomic_write_json(path, {"complete": True})

    assert json.loads(path.read_text()) == {"complete": True}
    assert not (tmp_path / "result.json.tmp").exists()


def test_token_digest_changes_when_order_or_boundaries_change():
    baseline = token_rows_digest([[1, 2], [3]])

    assert baseline != token_rows_digest([[3], [1, 2]])
    assert baseline != token_rows_digest([[1], [2, 3]])


def test_calibration_and_evaluation_token_sets_must_be_disjoint():
    with pytest.raises(ValueError, match="overlap"):
        validate_disjoint("same", "same")

    validate_disjoint("calibration", "evaluation")


def test_partial_token_window_overlap_is_rejected():
    with pytest.raises(ValueError, match="overlap"):
        validate_no_row_overlap(["cal-a", "shared"], ["shared", "eval-b"])

    validate_no_row_overlap(["cal-a"], ["eval-b"])


def test_protocol_contract_requires_exact_match():
    result = {"protocol_contract": {"input": "sha", "prompts": 128}}

    validate_protocol_contract(result, {"input": "sha", "prompts": 128})
    with pytest.raises(RuntimeError, match="protocol"):
        validate_protocol_contract(result, {"input": "other", "prompts": 128})
