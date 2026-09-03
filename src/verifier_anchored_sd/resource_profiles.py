"""Hardware-aware fitting profiles for E0 after LLM calibration capture.

The important distinction is between *capture* residency and *fit* residency.  A
16GB card may need CPU offload while Qwen3-4B and Qwen3-1.7B are simultaneously
loaded for cache capture, but E0 deletes both models before fitting.  The expensive
R² selector and ridge normal equations should therefore reuse the freed CUDA device
instead of falling back to CPU.
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


def e0_fit_profile(
    memory_profile: MemoryProfile,
    *,
    head_mode: HeadMode,
    draft_layers: int,
) -> E0FitProfile:
    """Return conservative CUDA block sizes for the post-capture E0 fit.

    ``selection_layer_block=draft_layers`` deliberately makes layer selection a
    single pass over calibration shards.  Once the LLM weights have been released,
    the selector's per-layer statistics fit comfortably on 16GB for the Qwen3
    4B->1.7B matched-KV pair.  The final full-head k=8 ridge has much larger
    sufficient statistics, so it uses smaller target-layer blocks than the
    matched-head variant.
    """
    if draft_layers <= 0:
        raise ValueError("draft_layers must be positive")
    if memory_profile not in {"16gb", "24gb"}:
        raise ValueError("memory_profile must be '16gb' or '24gb'")
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
