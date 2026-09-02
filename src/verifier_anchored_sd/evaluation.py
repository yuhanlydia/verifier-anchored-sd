"""Small, deterministic evaluation helpers shared by benchmark scripts."""

from __future__ import annotations


def block_bucket_overlap(*, cursor: int, emitted: int, lo: int, hi: int) -> int:
    """Count emitted output positions that fall in the inclusive bucket [lo, hi].

    ``cursor`` is the number of output tokens emitted before the block, so the
    block occupies one-indexed positions ``cursor+1 .. cursor+emitted``.
    """
    if cursor < 0 or emitted < 0:
        raise ValueError("cursor and emitted must be non-negative")
    if lo < 1 or hi < lo:
        raise ValueError("bucket must be a non-empty positive inclusive interval")
    if emitted == 0:
        return 0
    start = max(cursor + 1, lo)
    end = min(cursor + emitted, hi)
    return max(0, end - start + 1)
