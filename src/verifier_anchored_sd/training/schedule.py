"""Small helpers that make optimizer-step semantics explicit and testable."""

from __future__ import annotations

from collections.abc import Iterator


def optimizer_microbatch_schedule(
    optimizer_steps: int,
    grad_accum: int,
) -> Iterator[tuple[int, int]]:
    """Yield ``(optimizer_step, microbatch)`` for exact gradient accumulation.

    ``optimizer_steps`` always means parameter updates, never microbatches.  This
    prevents a common silent under-training bug where ``steps=500, accum=16``
    performs only about 32 optimizer updates.
    """
    if optimizer_steps <= 0 or grad_accum <= 0:
        raise ValueError("optimizer_steps and grad_accum must be positive")
    for optimizer_step in range(1, optimizer_steps + 1):
        for microbatch in range(1, grad_accum + 1):
            yield optimizer_step, microbatch
