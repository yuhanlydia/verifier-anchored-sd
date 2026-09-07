"""Hard provenance gate between pair screening and E2 evaluation."""

from __future__ import annotations


def validate_e2_artifacts(
    screen_result: dict,
    mapper_metadata: dict,
    *,
    mapper_sha256: str,
    mapper_metadata_sha256: str,
) -> dict[str, str]:
    """Validate that E2 uses the exact mapper, pair, and revisions that passed."""
    if screen_result.get("gate", {}).get("status") != "pass":
        raise RuntimeError("pair screen must pass before E2")
    requested = screen_result.get("requested_rows")
    completed = screen_result.get("completed_rows")
    if not isinstance(requested, int) or requested <= 0 or completed != requested:
        raise RuntimeError("pair screen must be complete before E2")
    if screen_result.get("pair") != mapper_metadata.get("pair"):
        raise RuntimeError("screen result and mapper describe different pairs")
    if screen_result.get("source_model") != mapper_metadata.get("source_model"):
        raise RuntimeError("screen verifier revision differs from mapper metadata")
    if screen_result.get("draft_model") != mapper_metadata.get("draft_model"):
        raise RuntimeError("screen draft revision differs from mapper metadata")
    dtype = mapper_metadata.get("dtype")
    if not dtype or screen_result.get("protocol_contract", {}).get("dtype") != dtype:
        raise RuntimeError("screen dtype differs from mapper metadata")
    expected_sha = mapper_metadata.get("checkpoint_sha256")
    if not expected_sha or mapper_sha256 != expected_sha:
        raise RuntimeError("mapper checkpoint digest differs from mapper metadata")
    if screen_result.get("mapper_checkpoint_sha256") != mapper_sha256:
        raise RuntimeError("screen result is not bound to this mapper checkpoint")
    if screen_result.get("mapper_metadata_sha256") != mapper_metadata_sha256:
        raise RuntimeError("screen result is not bound to this mapper metadata")
    source = mapper_metadata["source_model"]
    draft = mapper_metadata["draft_model"]
    if source.get("tokenizer_hash") != draft.get("tokenizer_hash"):
        raise RuntimeError("mapper model tokenizer contracts differ")
    return {
        "target_revision": str(source["revision"]),
        "draft_revision": str(draft["revision"]),
        "tokenizer_hash": str(source["tokenizer_hash"]),
        "dtype": str(dtype),
    }
