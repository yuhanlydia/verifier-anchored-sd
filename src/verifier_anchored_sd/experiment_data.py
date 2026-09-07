"""Deterministic token-window construction for experiment inputs."""

from __future__ import annotations


def token_windows(tokenizer, texts, *, seq_len: int, count: int):
    """Yield non-overlapping fixed-length chunks, using all of each document."""
    if seq_len <= 0 or count <= 0:
        raise ValueError("seq_len and count must be positive")
    remaining = count
    for text in texts:
        ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        for start in range(0, ids.numel() - seq_len + 1, seq_len):
            yield ids[start : start + seq_len]
            remaining -= 1
            if remaining == 0:
                return
