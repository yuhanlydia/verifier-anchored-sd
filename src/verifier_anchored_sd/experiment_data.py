"""Deterministic token-window construction for experiment inputs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenWindow:
    token_ids: object
    document_id: int
    chunk_id: int


def token_windows_with_sources(tokenizer, texts, *, seq_len: int, count: int):
    """Yield disjoint token windows with their source-document provenance."""
    if seq_len <= 0 or count <= 0:
        raise ValueError("seq_len and count must be positive")
    remaining = count
    for document_id, text in enumerate(texts):
        ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        for chunk_id, start in enumerate(range(0, ids.numel() - seq_len + 1, seq_len)):
            yield TokenWindow(ids[start : start + seq_len], document_id, chunk_id)
            remaining -= 1
            if remaining == 0:
                return


def token_windows(tokenizer, texts, *, seq_len: int, count: int):
    """Yield non-overlapping fixed-length chunks, using all of each document."""
    for window in token_windows_with_sources(
        tokenizer, texts, seq_len=seq_len, count=count
    ):
        yield window.token_ids


def collect_tokenized_prompts(tokenizer, texts, *, max_tokens: int, count: int) -> list[list[int]]:
    """Collect exactly ``count`` valid prompts, skipping short source records."""
    if max_tokens < 2 or count <= 0:
        raise ValueError("max_tokens must exceed one and count must be positive")
    rows: list[list[int]] = []
    for text in texts:
        ids = tokenizer(text, add_special_tokens=True, return_tensors="pt")["input_ids"][
            0, :max_tokens
        ]
        if ids.numel() >= 2:
            rows.append([int(token) for token in ids.tolist()])
        if len(rows) == count:
            return rows
    raise RuntimeError(f"only {len(rows)}/{count} valid prompts were available")
