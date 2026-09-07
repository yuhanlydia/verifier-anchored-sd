from pathlib import Path

import pytest
import torch

from verifier_anchored_sd.distribution_artifacts import (
    exact_probability_paths,
    load_probability_shard,
    save_probability_shard,
)


def test_probability_shard_round_trips_fp32(tmp_path: Path):
    path = tmp_path / "00000.pt"
    probs = torch.tensor([0.25, 0.75], dtype=torch.float64)
    metadata = {"sequence_id": "00000", "next_token_id": 1, "token_digest": "abc"}

    save_probability_shard(path, probs, metadata)
    loaded, loaded_metadata = load_probability_shard(path)

    assert loaded.dtype == torch.float32
    assert torch.allclose(loaded, torch.tensor([0.25, 0.75]))
    assert loaded_metadata == metadata


def test_probability_shard_rejects_non_normalized_rows(tmp_path: Path):
    with pytest.raises(ValueError, match="sum to one"):
        save_probability_shard(
            tmp_path / "bad.pt",
            torch.tensor([0.25, 0.25]),
            {"sequence_id": "00000"},
        )


def test_probability_shard_rejects_nonfinite_values(tmp_path: Path):
    with pytest.raises(ValueError, match="finite"):
        save_probability_shard(
            tmp_path / "bad.pt",
            torch.tensor([float("nan"), 1.0]),
            {"sequence_id": "00000"},
        )


def test_exact_probability_paths_requires_exact_set(tmp_path: Path):
    root = tmp_path / "target_probs"
    root.mkdir()
    for index in range(2):
        save_probability_shard(
            root / f"{index:05d}.pt",
            torch.tensor([0.5, 0.5]),
            {"sequence_id": f"{index:05d}"},
        )

    assert exact_probability_paths(root, 2) == [root / "00000.pt", root / "00001.pt"]

    save_probability_shard(
        root / "extra.pt", torch.tensor([0.5, 0.5]), {"sequence_id": "extra"}
    )
    with pytest.raises(RuntimeError, match="shard set mismatch"):
        exact_probability_paths(root, 2)
