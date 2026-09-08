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


def validate_probability_binding(
    cache_metadata: dict,
    probability_metadata: dict,
    probabilities: torch.Tensor,
    *,
    expected_vocab_size: int,
) -> None:
    """Bind a verifier distribution to the exact cache/token row it accompanies."""
    for key in ("sequence_id", "token_digest", "next_token_id", "model_revision", "dtype"):
        if cache_metadata.get(key) != probability_metadata.get(key):
            raise RuntimeError(f"probability/cache {key} mismatch")
    if probability_metadata.get("storage_dtype") != "float32":
        raise RuntimeError("probability storage_dtype must be float32")
    if float(probability_metadata.get("temperature", float("nan"))) != 1.0:
        raise RuntimeError("probability temperature must be 1.0")
    if expected_vocab_size <= 0:
        raise ValueError("expected_vocab_size must be positive")
    metadata_vocab = probability_metadata.get("vocab_size")
    if metadata_vocab is None or int(metadata_vocab) != expected_vocab_size:
        raise RuntimeError("probability metadata vocab_size mismatch")
    if probabilities.ndim != 1 or probabilities.numel() != expected_vocab_size:
        raise RuntimeError("probability tensor vocab size mismatch")
    try:
        _validate_probabilities(probabilities)
    except ValueError as exc:
        raise RuntimeError(f"invalid bound probability vector: {exc}") from exc


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
