"""Runtime mapper loading shared by ridge, paper-MLP, and subspace experiments."""

from __future__ import annotations

import json
from pathlib import Path

from .paper_mapper_baselines import GroupedHeadMLPMapper
from .spec_decode.target_to_draft_mapper import RidgeKVMapper


def load_runtime_mapper(
    checkpoint: str | Path,
    metadata_path: str | Path | None = None,
    *,
    map_location: str = "cpu",
):
    """Load the mapper implementation declared by the provenance metadata.

    Old ridge metadata did not carry an explicit ``kind``; those artifacts remain
    backward compatible and default to ``RidgeKVMapper``.  Paper MLP artifacts must
    explicitly declare ``mapper.kind = mlp_full_head`` so a large nonlinear
    checkpoint can never be mistaken for a ridge artifact.
    """
    checkpoint = Path(checkpoint)
    metadata_path = Path(metadata_path or f"{checkpoint}.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    mapper_config = metadata.get("mapper", {})
    if not isinstance(mapper_config, dict):
        raise RuntimeError("mapper metadata field must be an object")
    kind = str(mapper_config.get("kind", "ridge"))
    if kind == "mlp_full_head":
        return GroupedHeadMLPMapper.load(checkpoint, map_location=map_location)
    if kind.startswith("ridge") or kind in {"ridge", "matched", "full"}:
        return RidgeKVMapper.load(checkpoint, map_location=map_location)
    raise RuntimeError(f"unsupported mapper kind: {kind}")
