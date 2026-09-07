"""On-disk contracts for sequential KV-cache capture and fitting."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .experiment_artifacts import atomic_write_json
from .spec_decode.cache_state import CacheState, LayerKV, RotaryFactors


def _cache_payload(cache: CacheState, metadata: dict) -> dict:
    return {
        "schema_version": 1,
        "metadata": dict(metadata),
        "keys": [layer.key.detach().cpu() for layer in cache.layers],
        "values": [layer.value.detach().cpu() for layer in cache.layers],
        "keys_are_content": cache.keys_are_content,
        "rotary_cos": None if cache.rotary is None else cache.rotary.cos.detach().cpu(),
        "rotary_sin": None if cache.rotary is None else cache.rotary.sin.detach().cpu(),
        "rotary_interleaved": None if cache.rotary is None else cache.rotary.interleaved,
    }


def save_cache_shard(path: str | Path, cache: CacheState, metadata: dict) -> None:
    """Atomically save a plain-tensor cache shard."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(_cache_payload(cache, metadata), temporary)
    temporary.replace(destination)


def save_token_rows(path: str | Path, rows: list[list[int]], metadata: dict) -> None:
    """Atomically save frozen token windows without pickled custom classes."""
    normalized = [[int(token) for token in row] for row in rows]
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(
        {"schema_version": 1, "metadata": dict(metadata), "rows": normalized},
        temporary,
    )
    temporary.replace(destination)


def load_token_rows(path: str | Path) -> tuple[list[list[int]], dict]:
    """Load frozen token windows and require integer row structure."""
    source = Path(path)
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(f"failed to load token rows {source}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise RuntimeError(f"unsupported token-row schema: {source}")
    rows, metadata = payload.get("rows"), payload.get("metadata")
    if (
        not isinstance(rows, list)
        or not all(isinstance(row, list) for row in rows)
        or not all(isinstance(token, int) for row in rows for token in row)
        or not isinstance(metadata, dict)
    ):
        raise RuntimeError(f"invalid token-row payload: {source}")
    return rows, metadata


def load_cache_shard(path: str | Path) -> tuple[CacheState, dict]:
    """Load and validate a cache shard saved by :func:`save_cache_shard`."""
    source = Path(path)
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(f"failed to load cache shard {source}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise RuntimeError(f"unsupported cache shard schema: {source}")
    keys, values = payload.get("keys"), payload.get("values")
    if not isinstance(keys, list) or not isinstance(values, list) or len(keys) != len(values):
        raise RuntimeError(f"invalid cache tensor lists: {source}")
    rotary = None
    cos, sin = payload.get("rotary_cos"), payload.get("rotary_sin")
    if (cos is None) != (sin is None):
        raise RuntimeError(f"incomplete rotary factors: {source}")
    if cos is not None:
        rotary = RotaryFactors(cos, sin, bool(payload.get("rotary_interleaved", False)))
    cache = CacheState(
        (LayerKV(key, value) for key, value in zip(keys, values, strict=True)),
        rotary=rotary,
        keys_are_content=bool(payload.get("keys_are_content", False)),
    )
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise TypeError(f"cache shard metadata is missing: {source}")
    return cache, metadata


def sample_cache_tokens(cache: CacheState, stride: int) -> CacheState:
    """Sample identical token positions from cache tensors and RoPE factors."""
    if stride <= 0:
        raise ValueError("stride must be positive")
    rotary = None
    if cache.rotary is not None:
        rotary = RotaryFactors(
            cache.rotary.cos[..., ::stride, :],
            cache.rotary.sin[..., ::stride, :],
            cache.rotary.interleaved,
        )
    return CacheState(
        (
            LayerKV(layer.key[..., ::stride, :], layer.value[..., ::stride, :])
            for layer in cache.layers
        ),
        rotary=rotary,
        keys_are_content=cache.keys_are_content,
    )


def write_or_validate_manifest(directory: str | Path, contract: dict) -> Path:
    """Create a frozen manifest or require exact equality with the existing one."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "manifest.json"
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid existing manifest: {path}") from exc
        if existing != contract:
            raise RuntimeError(
                f"artifact directory contains a different contract: {path}"
            )
    else:
        atomic_write_json(path, contract)
    return path


def validate_capture_pair(source: dict, draft: dict) -> dict[str, int]:
    """Validate that two sequential captures form one directional mapper dataset."""
    if source.get("role") != "source" or draft.get("role") != "draft":
        raise ValueError("capture roles must be source followed by draft")
    if source.get("pair") != draft.get("pair"):
        raise ValueError("source and draft captures describe different model pairs")
    if source.get("capture") != draft.get("capture"):
        raise ValueError("source and draft captures use different capture parameters")
    if source.get("token_rows_digest") != draft.get("token_rows_digest"):
        raise ValueError("source and draft captures use different token windows")
    source_model, draft_model = source.get("model", {}), draft.get("model", {})
    if source_model.get("tokenizer_hash") != draft_model.get("tokenizer_hash"):
        raise ValueError("source and draft tokenizer contracts differ")
    for field in ("kv_heads", "head_dim"):
        if source_model.get(field) != draft_model.get(field):
            raise ValueError(f"source and draft {field} differ")
    pair = source["pair"]
    if source_model.get("id") != pair.get("target") or draft_model.get("id") != pair.get(
        "draft"
    ):
        raise ValueError("capture model IDs do not match the directional pair")
    return {
        "target_layers": int(source_model["layers"]),
        "draft_layers": int(draft_model["layers"]),
        "kv_heads": int(source_model["kv_heads"]),
        "head_dim": int(source_model["head_dim"]),
    }
