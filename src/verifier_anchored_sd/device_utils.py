"""Small device-selection helpers for mixed CPU/GPU model placement."""

from __future__ import annotations

from collections.abc import Iterable

import torch


def timing_device_from_parameter_devices(devices: Iterable[str | torch.device]) -> torch.device:
    """Prefer the first CUDA shard; otherwise return CPU.

    Accelerate/device-map offload can make ``next(model.parameters()).device`` be
    CPU even though most timed work launches CUDA kernels.  Benchmark timers must
    synchronize the actual GPU in that case.
    """
    resolved = [torch.device(device) for device in devices]
    for device in resolved:
        if device.type == "cuda":
            return device
    return torch.device("cpu")


def timing_device_for_models(*models) -> torch.device:
    """Resolve the CUDA timing device across one or more potentially sharded models."""
    return timing_device_from_parameter_devices(
        parameter.device for model in models for parameter in model.parameters()
    )
