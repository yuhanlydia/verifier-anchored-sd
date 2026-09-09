"""Contracts shared by paper-faithful full-head ridge and nonlinear MLP fitting."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MLPTrainingProfile:
    hidden_dim: int
    epochs: int
    batch_size: int
    learning_rate: float
    optimizer: str
    loss: str
    paper_faithful: bool


def paper_mlp_profile(name: str) -> MLPTrainingProfile:
    """Return an explicit training profile; never silently downgrade the paper run."""
    if name == "paper":
        # Appendix E of arXiv:2608.03893.
        return MLPTrainingProfile(
            hidden_dim=1024,
            epochs=20,
            batch_size=4096,
            learning_rate=1e-3,
            optimizer="adam",
            loss="mse",
            paper_faithful=True,
        )
    if name == "pilot":
        # Same nonlinear architecture and optimizer, fewer updates for a capacity
        # diagnostic on a single 32/48GB host.  Results must not be labeled paper reproduction.
        return MLPTrainingProfile(
            hidden_dim=1024,
            epochs=2,
            batch_size=1024,
            learning_rate=1e-3,
            optimizer="adam",
            loss="mse",
            paper_faithful=False,
        )
    raise ValueError("MLP profile must be 'paper' or 'pilot'")


def validate_mlp_ridge_metadata(
    ridge_metadata: dict,
    *,
    pair: dict,
    target_revision: str,
    draft_revision: str,
    draft_layers: int,
) -> list[list[int]]:
    """Require MLP and ridge to use the same full-head layer-selection support."""
    if ridge_metadata.get("pair") != pair:
        raise RuntimeError("MLP ridge-selection metadata describes a different pair")
    if ridge_metadata.get("source_model", {}).get("revision") != target_revision:
        raise RuntimeError("MLP ridge-selection verifier revision differs")
    if ridge_metadata.get("draft_model", {}).get("revision") != draft_revision:
        raise RuntimeError("MLP ridge-selection draft revision differs")
    mapper = ridge_metadata.get("mapper")
    if not isinstance(mapper, dict):
        raise RuntimeError("MLP ridge-selection metadata lacks mapper contract")
    if mapper.get("head_mode") != "full" or mapper.get("kind") not in {
        "ridge_full_head",
        "ridge_full_head_kvbridge",
    }:
        raise RuntimeError("paper MLP requires full-head ridge layer selection")
    selected = mapper.get("selected_layers")
    if (
        not isinstance(selected, list)
        or len(selected) != draft_layers
        or any(not isinstance(row, list) or not row for row in selected)
    ):
        raise RuntimeError("full-head ridge metadata has invalid selected layers")
    return [[int(index) for index in row] for row in selected]
