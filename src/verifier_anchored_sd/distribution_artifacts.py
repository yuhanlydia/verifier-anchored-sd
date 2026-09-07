"""On-disk contracts for exact verifier next-token distributions."""

from __future__ import annotations

from pathlib import Path

import torch


def _validate_probabilities(probabilities: torch.Tensor) -> torch.Tensor:
    probs = probabilities.detach().float().cpu()
    if probs.ndim != 1:
        raise ValueError("probability shard must contain one [vocab] vector")
    if probs.numel() == 0:
        raise ValueError("probability shard vocabulary must be non-empty")
    if not torch.isfinite(probs).all():
        raise ValueError("probabilities must be finite")
    if (probs < 0).any():
        raise ValueError("probabilities must be non-negative")
    total = probs.sum()
    if not torch.allclose(total, torch.tensor(1.0), atol=1e-5, rtol=1e-5):
        raise ValueError("probabilities must sum to one")
    return probs


def save_probability_shard(
    path: str | Path,
    probabilities: torch.Tensor,
    metadata: dict,
) -> None:
    """Atomically save one FP32 verifier next-token distribution."""
    probs = _validate_probabilities(probabilities)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(
        {
            "schema_version": 1,
            "metadata": dict(metadata),
            "probabilities": probs,
        },
        temporary,
    )
    temporary.replace(destination)


def load_probability_shard(path: str | Path) -> tuple[torch.Tensor, dict]:
    """Load and validate a probability shard produced by this module."""
    source = Path(path)
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(f"failed to load probability shard {source}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise RuntimeError(f"unsupported probability shard schema: {source}")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise RuntimeError(f"probability shard metadata is missing: {source}")
    probabilities = payload.get("probabilities")
    if not isinstance(probabilities, torch.Tensor):
        raise RuntimeError(f"probability shard tensor is missing: {source}")
    try:
        probabilities = _validate_probabilities(probabilities)
    except ValueError as exc:
        raise RuntimeError(f"invalid probability shard {source}: {exc}") from exc
    return probabilities, metadata


def exact_probability_paths(directory: str | Path, count: int) -> list[Path]:
    """Return ordered probability shards only for an exact zero-padded set."""
    if count <= 0:
        raise ValueError("shard count must be positive")
    root = Path(directory)
    expected = [root / f"{index:05d}.pt" for index in range(count)]
    actual = sorted(root.glob("*.pt")) if root.exists() else []
    if actual != expected:
        missing = [str(path) for path in expected if path not in actual]
        extra = [str(path) for path in actual if path not in expected]
        raise RuntimeError(
            f"probability shard set mismatch; missing={missing}, extra={extra}"
        )
    return expected
