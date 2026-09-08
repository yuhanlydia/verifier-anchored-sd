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
        "mapper_metadata_sha256": "metadata-sha",
        "requested_rows": 128,
        "completed_rows": 128,
        "native_fidelity": {"gate": {"status": "fail"}},
        "target_alignment": {"gate": {"status": "support"}},
        "decision": {"status": "go_sd"},
        "protocol_contract": {"dtype": "bfloat16"},
    }
    return screen, metadata


def test_e2_contract_binds_screen_to_mapper_and_revisions():
    screen, metadata = _artifacts()

    contract = validate_e2_artifacts(
        screen,
        metadata,
        mapper_sha256="mapper-sha",
        mapper_metadata_sha256="metadata-sha",
    )

    assert contract["target_revision"] == "t-rev"
    assert contract["draft_revision"] == "d-rev"
    assert contract["tokenizer_hash"] == "tok"
    assert contract["dtype"] == "bfloat16"


def test_e2_contract_allows_target_support_even_when_native_fidelity_failed():
    screen, metadata = _artifacts()
    assert screen["native_fidelity"]["gate"]["status"] == "fail"

    validate_e2_artifacts(
        screen,
        metadata,
        mapper_sha256="mapper-sha",
        mapper_metadata_sha256="metadata-sha",
    )


def test_e2_contract_rejects_target_alignment_harm():
    screen, metadata = _artifacts()
    screen["target_alignment"]["gate"]["status"] = "harm"
    screen["decision"]["status"] = "stop_pair"

    with pytest.raises(RuntimeError, match="target alignment"):
        validate_e2_artifacts(
            screen,
            metadata,
            mapper_sha256="mapper-sha",
            mapper_metadata_sha256="metadata-sha",
        )


def test_e2_contract_rejects_legacy_screen_without_target_alignment_decision():
    screen, metadata = _artifacts()
    screen.pop("decision")

    with pytest.raises(RuntimeError, match="target alignment"):
        validate_e2_artifacts(
            screen,
            metadata,
            mapper_sha256="mapper-sha",
            mapper_metadata_sha256="metadata-sha",
        )


def test_e2_contract_rejects_unrelated_mapper():
    screen, metadata = _artifacts()

    with pytest.raises(RuntimeError, match="checkpoint"):
        validate_e2_artifacts(
            screen,
            metadata,
            mapper_sha256="different",
            mapper_metadata_sha256="metadata-sha",
        )


def test_e2_contract_rejects_incomplete_screen():
    screen, metadata = _artifacts()
    screen["completed_rows"] = 127

    with pytest.raises(RuntimeError, match="complete"):
        validate_e2_artifacts(
            screen,
            metadata,
            mapper_sha256="mapper-sha",
            mapper_metadata_sha256="metadata-sha",
        )


def test_e2_contract_rejects_edited_mapper_metadata():
    screen, metadata = _artifacts()

    with pytest.raises(RuntimeError, match="metadata"):
        validate_e2_artifacts(
            screen,
            metadata,
            mapper_sha256="mapper-sha",
            mapper_metadata_sha256="edited",
        )
