import pytest
import torch

from verifier_anchored_sd.cache_artifacts import (
    exact_shard_paths,
    load_cache_shard,
    load_token_rows,
    sample_cache_tokens,
    save_cache_shard,
    save_token_rows,
    write_or_validate_manifest,
)
from verifier_anchored_sd.spec_decode.cache_state import CacheState, LayerKV, RotaryFactors


def cache_with_rotary(tokens: int = 5) -> CacheState:
    key = torch.arange(tokens * 4, dtype=torch.float32).reshape(1, 1, tokens, 4)
    value = key + 100
    cos = torch.arange(tokens * 4, dtype=torch.float32).reshape(1, tokens, 4)
    sin = cos + 200
    return CacheState(
        [LayerKV(key, value)],
        rotary=RotaryFactors(cos, sin, interleaved=False),
        keys_are_content=False,
    )


def test_cache_shard_round_trip_preserves_rotary_and_metadata(tmp_path):
    original = cache_with_rotary(tokens=3)
    path = tmp_path / "00000.pt"

    save_cache_shard(path, original, {"sequence_id": "00000", "token_digest": "abc"})
    restored, metadata = load_cache_shard(path)

    assert metadata == {"sequence_id": "00000", "token_digest": "abc"}
    assert restored.keys_are_content == original.keys_are_content
    assert torch.equal(restored.layers[0].key, original.layers[0].key)
    assert torch.equal(restored.layers[0].value, original.layers[0].value)
    assert torch.equal(restored.rotary.cos, original.rotary.cos)
    assert torch.equal(restored.rotary.sin, original.rotary.sin)


def test_cache_sampling_keeps_kv_and_rotary_positions_aligned():
    sampled = sample_cache_tokens(cache_with_rotary(tokens=5), stride=2)

    assert sampled.seq_len == 3
    assert torch.equal(sampled.layers[0].key, cache_with_rotary().layers[0].key[..., ::2, :])
    assert torch.equal(sampled.rotary.cos, cache_with_rotary().rotary.cos[..., ::2, :])


def test_existing_manifest_rejects_changed_contract(tmp_path):
    write_or_validate_manifest(tmp_path, {"model": "Qwen/A", "revision": "a"})

    with pytest.raises(RuntimeError, match="different contract"):
        write_or_validate_manifest(tmp_path, {"model": "Qwen/A", "revision": "b"})


def test_token_rows_round_trip_as_plain_integer_lists(tmp_path):
    path = tmp_path / "tokens.pt"

    save_token_rows(path, [[1, 2, 3], [4, 5, 6]], {"count": 2, "seq_len": 3})
    rows, metadata = load_token_rows(path)

    assert rows == [[1, 2, 3], [4, 5, 6]]
    assert metadata == {"count": 2, "seq_len": 3}


def test_exact_shard_set_rejects_missing_or_extra_files(tmp_path):
    (tmp_path / "00000.pt").touch()
    with pytest.raises(RuntimeError, match="shard set"):
        exact_shard_paths(tmp_path, count=2)

    (tmp_path / "00001.pt").touch()
    assert exact_shard_paths(tmp_path, count=2) == [
        tmp_path / "00000.pt",
        tmp_path / "00001.pt",
    ]

    (tmp_path / "00002.pt").touch()
    with pytest.raises(RuntimeError, match="shard set"):
        exact_shard_paths(tmp_path, count=2)
