import pytest

from verifier_anchored_sd.paper_mapper_fit import (
    paper_mlp_profile,
    validate_mlp_ridge_metadata,
)


def test_paper_mlp_profile_matches_reported_architecture_and_optimizer():
    profile = paper_mlp_profile("paper")

    assert profile.hidden_dim == 1024
    assert profile.epochs == 20
    assert profile.batch_size == 4096
    assert profile.learning_rate == pytest.approx(1e-3)
    assert profile.optimizer == "adam"
    assert profile.loss == "mse"
    assert profile.paper_faithful is True


def test_pilot_profile_keeps_architecture_but_is_explicitly_non_paper():
    profile = paper_mlp_profile("pilot")

    assert profile.hidden_dim == 1024
    assert profile.epochs < 20
    assert profile.paper_faithful is False


def test_mlp_requires_full_head_ridge_layer_selection_from_same_pair():
    ridge = {
        "pair": {"target": "T", "draft": "D"},
        "source_model": {"revision": "t"},
        "draft_model": {"revision": "d"},
        "mapper": {
            "kind": "ridge_full_head",
            "head_mode": "full",
            "selected_layers": [[0, 1], [1, 2]],
        },
    }

    selected = validate_mlp_ridge_metadata(
        ridge,
        pair={"target": "T", "draft": "D"},
        target_revision="t",
        draft_revision="d",
        draft_layers=2,
    )

    assert selected == [[0, 1], [1, 2]]


def test_mlp_rejects_matched_head_selection_metadata():
    ridge = {
        "pair": {"target": "T", "draft": "D"},
        "source_model": {"revision": "t"},
        "draft_model": {"revision": "d"},
        "mapper": {
            "kind": "ridge_matched_head",
            "head_mode": "matched",
            "selected_layers": [[0]],
        },
    }

    with pytest.raises(RuntimeError, match="full-head"):
        validate_mlp_ridge_metadata(
            ridge,
            pair={"target": "T", "draft": "D"},
            target_revision="t",
            draft_revision="d",
            draft_layers=1,
        )
