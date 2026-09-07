"""Stable tokenizer and model geometry contracts for cross-model cache reuse."""

from __future__ import annotations

import hashlib
import json


def tokenizer_contract(tokenizer) -> dict:
    """Return a JSON-safe contract covering every token ID and special token."""
    return {
        "vocab": sorted((str(token), int(index)) for token, index in tokenizer.get_vocab().items()),
        "special_tokens_map": {
            str(name): str(value)
            for name, value in sorted(tokenizer.special_tokens_map.items())
        },
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }


def tokenizer_contract_hash(tokenizer) -> str:
    payload = json.dumps(
        tokenizer_contract(tokenizer),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_tokenizer_pair(source_tokenizer, draft_tokenizer) -> str:
    """Require identical token-to-ID and special-token behavior."""
    source = tokenizer_contract(source_tokenizer)
    draft = tokenizer_contract(draft_tokenizer)
    if source != draft:
        raise ValueError(
            "target and draft tokenizer contracts differ; cross-model cache positions/token IDs are unsafe"
        )
    return tokenizer_contract_hash(source_tokenizer)


def model_metadata(model, model_id: str, tokenizer_hash: str) -> dict:
    """Capture the frozen model revision and KV geometry used by an artifact."""
    config = getattr(model.config, "text_config", model.config)
    attention_heads = int(config.num_attention_heads)
    return {
        "id": model_id,
        "revision": str(getattr(model.config, "_commit_hash", None) or "unresolved"),
        "tokenizer_hash": tokenizer_hash,
        "layers": int(config.num_hidden_layers),
        "kv_heads": int(getattr(config, "num_key_value_heads", attention_heads)),
        "head_dim": int(getattr(config, "head_dim", config.hidden_size // attention_heads)),
        "architecture": str(getattr(config, "architectures", [type(model).__name__])[0]),
    }
