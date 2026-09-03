"""Hardware-aware E0/E1/E2 profiles for 16GB and 24GB single-GPU experiments.

A key distinction is between *capture* residency and *fit* residency.  A 16GB card
may need CPU offload while Qwen3-4B and Qwen3-1.7B coexist for cache capture, but E0
deletes both models before fitting.  R² selection and ridge statistics should then
reuse the freed CUDA device.  Inference profiles keep batch=1 as the latency result
and explicitly sweep larger batches until the capacity boundary for throughput.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MemoryProfile = Literal["16gb", "24gb"]
HeadMode = Literal["full", "matched"]


@dataclass(frozen=True)
class E0FitProfile:
    accumulation_device: str
    selection_layer_block: int
    fit_layer_block: int


@dataclass(frozen=True)
class E2Profile:
    prompts: int
    prompt_tokens: int
    new_tokens: int
    gamma: int


def _validate_memory_profile(memory_profile: str) -> None:
    if memory_profile not in {"16gb", "24gb"}:
        raise ValueError("memory_profile must be '16gb' or '24gb'")


def e0_fit_profile(
    memory_profile: MemoryProfile,
    *,
    head_mode: HeadMode,
    draft_layers: int,
) -> E0FitProfile:
    """Return conservative CUDA block sizes for the post-capture E0 fit."""
    if draft_layers <= 0:
        raise ValueError("draft_layers must be positive")
    _validate_memory_profile(memory_profile)
    if head_mode not in {"full", "matched"}:
        raise ValueError("head_mode must be 'full' or 'matched'")

    if memory_profile == "16gb":
        fit_block = 2 if head_mode == "full" else 8
    else:
        fit_block = 4 if head_mode == "full" else 16

    return E0FitProfile(
        accumulation_device="cuda",
        selection_layer_block=draft_layers,
        fit_layer_block=min(fit_block, draft_layers),
    )


def e1_batch_sizes(memory_profile: MemoryProfile) -> list[int]:
    """Batch sweep used to find the real single-GPU throughput/capacity frontier."""
    _validate_memory_profile(memory_profile)
    return [1, 2, 4] if memory_profile == "16gb" else [1, 2, 4, 8]


def e2_profile(memory_profile: MemoryProfile) -> E2Profile:
    """Quality pilot sizes: many short 16GB samples, full long 24GB pilot."""
    _validate_memory_profile(memory_profile)
    if memory_profile == "16gb":
        # Acceptance estimation benefits more from independent prompts than from one
        # very long sample.  A separate drift script covers 256-token continuations.
        return E2Profile(prompts=64, prompt_tokens=512, new_tokens=64, gamma=4)
    return E2Profile(prompts=200, prompt_tokens=512, new_tokens=512, gamma=4)
