from pathlib import Path

import pytest
import torch

from verifier_anchored_sd.kv_subspace import SubspaceBasisArtifact
from verifier_anchored_sd.spec_decode.cache_state import CacheState, LayerKV, RotaryFactors
from verifier_anchored_sd.subspace_fit import (
    assert_gradient_coverage,
    cache_content_rows,
    cache_gradient_rows,
    make_gradient_cache,
    target_kl_loss,
    teacher_delta_rows,
    validate_subspace_fit_inputs,
)


def _mapper_metadata():
    source = {
        "id": "Qwen/T",
        "revision": "target-rev",
        "tokenizer_hash": "tok",
        "layers": 2,
        "kv_heads": 1,
        "head_dim": 2,
    }
    draft = {
        "id": "Qwen/D",
        "revision": "draft-rev",
        "tokenizer_hash": "tok",
        "layers": 2,
        "kv_heads": 1,
        "head_dim": 2,
    }
    return {
        "pair": {"target": "Qwen/T", "draft": "Qwen/D"},
        "source_model": source,
        "draft_model": draft,
        "dtype": "bfloat16",
        "checkpoint_sha256": "mapper-sha",
        "token_row_digests": ["cal-a", "cal-b"],
    }


def _screen_manifest():
    metadata = _mapper_metadata()
    return {
        "schema_version": 3,
        "role": "screen_source_with_distribution",
        "pair": metadata["pair"],
        "model": metadata["source_model"],
        "dtype": "bfloat16",
        "capture": {"count": 8, "prefix_tokens": 16, "stride": 1},
        "target_distribution": {
            "storage_dtype": "float32",
            "temperature": 1.0,
            "vocab_size": 7,
        },
        "input_sha256": "fit-input",
        "token_rows_digest": "fit-rows",
        "token_row_digests": ["fit-a", "fit-b"],
        "window_sources": [{"document_id": 1, "chunk_id": 0}, {"document_id": 2, "chunk_id": 0}],
    }


def _identity_rotary(tokens: int, dim: int) -> RotaryFactors:
    return RotaryFactors(torch.ones(tokens, dim), torch.zeros(tokens, dim), interleaved=False)


def _cache(key: torch.Tensor, value: torch.Tensor) -> CacheState:
    rotary = _identity_rotary(key.shape[-2], key.shape[-1])
    return CacheState([LayerKV(key.clone(), value.clone())], rotary=rotary, keys_are_content=False)


def test_fit_contract_binds_mapper_and_requires_distribution_screen():
    contract = validate_subspace_fit_inputs(
        _mapper_metadata(),
        _screen_manifest(),
        mapper_sha256="mapper-sha",
        requested_prompts=2,
    )

    assert contract["draft_layers"] == 2
    assert contract["kv_heads"] == 1
    assert contract["head_dim"] == 2
    assert contract["vocab_size"] == 7


def test_fit_contract_rejects_mapper_digest_mismatch():
    with pytest.raises(RuntimeError, match="mapper checkpoint"):
        validate_subspace_fit_inputs(
            _mapper_metadata(),
            _screen_manifest(),
            mapper_sha256="different",
            requested_prompts=2,
        )


def test_fit_contract_rejects_legacy_cache_only_screen():
    legacy = _screen_manifest()
    legacy.pop("target_distribution")
    legacy["schema_version"] = 2

    with pytest.raises(ValueError, match="target distribution"):
        validate_subspace_fit_inputs(
            _mapper_metadata(), legacy, mapper_sha256="mapper-sha", requested_prompts=2
        )


def test_fit_contract_rejects_mapper_calibration_overlap():
    overlap = _screen_manifest()
    overlap["token_row_digests"] = ["fit-a", "cal-b"]

    with pytest.raises(ValueError, match="overlap"):
        validate_subspace_fit_inputs(
            _mapper_metadata(), overlap, mapper_sha256="mapper-sha", requested_prompts=2
        )


def test_fit_contract_rejects_more_prompts_than_captured():
    with pytest.raises(ValueError, match="captured"):
        validate_subspace_fit_inputs(
            _mapper_metadata(), _screen_manifest(), mapper_sha256="mapper-sha", requested_prompts=9
        )


def test_gradient_cache_creates_leaf_kv_without_enabling_source_or_rotary_grad():
    source = _cache(
        torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]]),
        torch.tensor([[[[0.5, -1.0], [2.0, 1.0]]]]),
    )

    leaf = make_gradient_cache(source)

    assert leaf.layers[0].key.is_leaf and leaf.layers[0].key.requires_grad
    assert leaf.layers[0].value.is_leaf and leaf.layers[0].value.requires_grad
    assert source.layers[0].key.requires_grad is False
    assert source.layers[0].value.requires_grad is False
    assert leaf.rotary is not None
    assert leaf.rotary.cos.requires_grad is False


def test_cache_gradient_rows_are_layer_kind_head_token_dim_and_content_space():
    source = _cache(
        torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]]),
        torch.tensor([[[[0.5, -1.0], [2.0, 1.0]]]]),
    )
    leaf = make_gradient_cache(source)
    loss = 2.0 * leaf.layers[0].key.sum() + 3.0 * leaf.layers[0].value.sum()
    loss.backward()

    rows = cache_gradient_rows(leaf)

    assert rows.shape == (1, 2, 1, 2, 2)
    assert torch.allclose(rows[0, 0], torch.full((1, 2, 2), 2.0))
    assert torch.allclose(rows[0, 1], torch.full((1, 2, 2), 3.0))


def test_teacher_delta_rows_use_content_keys_and_value_space():
    native = _cache(
        torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]]),
        torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
    )
    mapped = _cache(
        torch.tensor([[[[2.0, 5.0], [1.0, 8.0]]]]),
        torch.tensor([[[[4.0, 1.0], [2.0, -1.0]]]]),
    )

    delta = teacher_delta_rows(native, mapped)
    mapped_rows = cache_content_rows(mapped)

    assert delta.shape == (1, 2, 1, 2, 2)
    assert torch.allclose(delta[0, 0, 0], torch.tensor([[1.0, 3.0], [-2.0, 4.0]]))
    assert torch.allclose(delta[0, 1, 0], torch.tensor([[3.0, 0.0], [1.0, -2.0]]))
    assert torch.allclose(mapped_rows[0, 0, 0], mapped.layers[0].key[0, 0])


def test_target_kl_loss_matches_hand_computed_distribution():
    logits = torch.log(torch.tensor([[[0.25, 0.75]]]))
    target = torch.tensor([0.5, 0.5])

    loss = target_kl_loss(logits, target)
    expected = 0.5 * torch.log(torch.tensor(0.5 / 0.25)) + 0.5 * torch.log(
        torch.tensor(0.5 / 0.75)
    )

    torch.testing.assert_close(loss, expected, atol=1e-6, rtol=1e-6)


def test_gradient_coverage_rejects_zero_and_nonfinite_layer_head_kind():
    good = torch.ones(2, 2, 3)
    assert_gradient_coverage(good, prefixes=4)

    zero = good.clone()
    zero[1, 0, 2] = 0
    with pytest.raises(RuntimeError, match="zero gradient"):
        assert_gradient_coverage(zero, prefixes=4)

    bad = good.clone()
    bad[0, 1, 0] = float("nan")
    with pytest.raises(RuntimeError, match="finite"):
        assert_gradient_coverage(bad, prefixes=4)


def test_basis_artifact_round_trip_preserves_provenance_and_tensors(tmp_path: Path):
    vectors = torch.eye(2).view(1, 1, 1, 2, 2).repeat(2, 2, 1, 1, 1)
    values = torch.tensor([2.0, 1.0]).view(1, 1, 1, 2).repeat(2, 2, 1, 1)
    ranks = torch.full((2, 2, 1), 2, dtype=torch.long)
    artifact = SubspaceBasisArtifact(
        metadata={
            "schema_version": 1,
            "draft_layers": 2,
            "kv_heads": 1,
            "head_dim": 2,
            "max_rank": 2,
            "mapper_checkpoint_sha256": "mapper-sha",
            "fit_token_rows_digest": "fit-rows",
        },
        bases={"grad": vectors},
        eigenvalues={"grad": values},
        valid_ranks={"grad": ranks},
    )
    path = tmp_path / "basis.pt"

    artifact.save(path)
    loaded = SubspaceBasisArtifact.load(path)

    assert loaded.metadata == artifact.metadata
    assert torch.equal(loaded.bases["grad"], vectors)
    assert torch.equal(loaded.eigenvalues["grad"], values)
    assert torch.equal(loaded.valid_ranks["grad"], ranks)
