import pytest

from verifier_anchored_sd.e2_contract import validate_e2_artifacts


def _artifacts():
    source = {"id": "T", "revision": "t-rev", "tokenizer_hash": "tok"}
    draft = {"id": "D", "revision": "d-rev", "tokenizer_hash": "tok"}
    metadata = {
        "pair": {"target": "T", "draft": "D"},
        "source_model": source,
        "draft_model": draft,
        "checkpoint_sha256": "mapper-sha",
        "dtype": "bfloat16",
    }
    screen = {
        "pair": metadata["pair"],
        "source_model": source,
        "draft_model": draft,
        "mapper_checkpoint_sha256": "mapper-sha",
        "requested_rows": 128,
        "completed_rows": 128,
        "gate": {"status": "pass"},
        "protocol_contract": {"dtype": "bfloat16"},
    }
    return screen, metadata


def test_e2_contract_binds_screen_to_mapper_and_revisions():
    screen, metadata = _artifacts()

    contract = validate_e2_artifacts(screen, metadata, mapper_sha256="mapper-sha")

    assert contract["target_revision"] == "t-rev"
    assert contract["draft_revision"] == "d-rev"
    assert contract["tokenizer_hash"] == "tok"
    assert contract["dtype"] == "bfloat16"


def test_e2_contract_rejects_unrelated_mapper():
    screen, metadata = _artifacts()

    with pytest.raises(RuntimeError, match="checkpoint"):
        validate_e2_artifacts(screen, metadata, mapper_sha256="different")


def test_e2_contract_rejects_incomplete_screen():
    screen, metadata = _artifacts()
    screen["completed_rows"] = 127

    with pytest.raises(RuntimeError, match="complete"):
        validate_e2_artifacts(screen, metadata, mapper_sha256="mapper-sha")
