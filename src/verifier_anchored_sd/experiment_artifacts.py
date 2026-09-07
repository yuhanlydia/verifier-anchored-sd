"""Reproducible artifact helpers for scientific experiment CLIs."""

from __future__ import annotations

import hashlib
import json
import os
import struct
from collections.abc import Iterable, Sequence
from pathlib import Path


def sha256_file(path: str | Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    """Hash a file without loading the complete artifact into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def token_rows_digest(rows: Iterable[Sequence[int]]) -> str:
    """Hash ordered token rows while preserving sequence boundaries."""
    digest = hashlib.sha256()
    for row in rows:
        digest.update(struct.pack("<Q", len(row)))
        for token in row:
            digest.update(struct.pack("<q", int(token)))
    return digest.hexdigest()


def atomic_write_json(path: str | Path, value) -> None:
    """Replace a JSON result only after its complete payload reaches disk."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(destination)


def validate_disjoint(calibration_digest: str, evaluation_digest: str) -> None:
    """Reject accidental reuse of the exact ordered window set."""
    if not calibration_digest or not evaluation_digest:
        raise ValueError("calibration and evaluation digests must be non-empty")
    if calibration_digest == evaluation_digest:
        raise ValueError("calibration/evaluation token-window overlap detected")
