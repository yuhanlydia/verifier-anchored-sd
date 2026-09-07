import json

import pytest

from verifier_anchored_sd.experiment_artifacts import (
    atomic_write_json,
    token_rows_digest,
    validate_disjoint,
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
