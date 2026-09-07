import pytest

from verifier_anchored_sd.cache_artifacts import validate_capture_pair
from verifier_anchored_sd.model_contracts import tokenizer_contract_hash, validate_tokenizer_pair


class FakeTokenizer:
    def __init__(self, vocab):
        self._vocab = vocab
        self.special_tokens_map = {"eos_token": "<eos>"}
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.pad_token_id = None

    def get_vocab(self):
        return self._vocab


def manifest(*, role: str, token_digest: str = "tokens", revision: str = "rev"):
    return {
        "schema_version": 1,
        "role": role,
        "pair": {"target": "Qwen/T", "draft": "Qwen/D"},
        "model": {
            "id": "Qwen/T" if role == "source" else "Qwen/D",
            "revision": revision,
            "tokenizer_hash": "tokenizer",
            "layers": 2 if role == "source" else 1,
            "kv_heads": 2,
            "head_dim": 4,
        },
        "capture": {"count": 3, "seq_len": 16, "stride": 2},
        "token_rows_digest": token_digest,
    }


def test_capture_pair_accepts_directional_layers_with_matching_kv_geometry():
    contract = validate_capture_pair(manifest(role="source"), manifest(role="draft"))

    assert contract == {"target_layers": 2, "draft_layers": 1, "kv_heads": 2, "head_dim": 4}


def test_capture_pair_rejects_different_token_windows():
    with pytest.raises(ValueError, match="token windows"):
        validate_capture_pair(
            manifest(role="source", token_digest="source"),
            manifest(role="draft", token_digest="draft"),
        )


def test_capture_pair_rejects_role_reversal():
    with pytest.raises(ValueError, match="roles"):
        validate_capture_pair(manifest(role="draft"), manifest(role="source"))


def test_tokenizer_hash_is_stable_across_vocab_insertion_order():
    left = FakeTokenizer({"b": 2, "a": 1})
    right = FakeTokenizer({"a": 1, "b": 2})

    assert tokenizer_contract_hash(left) == tokenizer_contract_hash(right)
    assert validate_tokenizer_pair(left, right) == tokenizer_contract_hash(left)


def test_tokenizer_pair_rejects_different_token_ids():
    with pytest.raises(ValueError, match="tokenizer contracts differ"):
        validate_tokenizer_pair(FakeTokenizer({"a": 1}), FakeTokenizer({"a": 2}))
