import json

import torch

from verifier_anchored_sd.mapper_io import load_runtime_mapper
from verifier_anchored_sd.paper_mapper_baselines import GroupedHeadMLPMapper, MLPMapperMetadata
from verifier_anchored_sd.spec_decode.target_to_draft_mapper import fit_ridge_mapper


def _tiny_ridge():
    observations = {}
    x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    y = torch.cat((x, x), dim=1)
    for kind in ("k", "v"):
        observations[(0, 0, kind)] = (x, y)
    return fit_ridge_mapper(
        observations,
        target_layers=1,
        draft_layers=1,
        kv_heads=1,
        head_dim=2,
        layer_selection=[[0]],
        lambda_=0.01,
        content_space=False,
        head_mode="matched",
    )


def _tiny_mlp():
    metadata = MLPMapperMetadata(
        target_layers=1,
        draft_layers=1,
        target_kv_heads=1,
        draft_kv_heads=1,
        head_dim=2,
        layer_selection=[[0]],
        content_space=True,
        hidden_dim=2,
    )
    shapes = {
        "w1": (1, 2, 1, 2, 2),
        "b1": (1, 2, 1, 2),
        "w2": (1, 2, 1, 2, 2),
        "b2": (1, 2, 1, 2),
        "w3": (1, 2, 1, 2, 2),
        "b3": (1, 2, 1, 2),
    }
    tensors = {name: torch.zeros(shape) for name, shape in shapes.items()}
    return GroupedHeadMLPMapper(metadata, **tensors)


def test_generic_loader_keeps_ridge_backward_compatible(tmp_path):
    mapper = _tiny_ridge()
    path = tmp_path / "ridge.pt"
    mapper.save(path)
    metadata = {"mapper": {"kind": "ridge_matched_head"}}
    meta_path = tmp_path / "ridge.pt.json"
    meta_path.write_text(json.dumps(metadata))

    loaded = load_runtime_mapper(path, meta_path)

    assert loaded.metadata.head_mode == "matched"


def test_generic_loader_loads_paper_mlp_from_metadata_kind(tmp_path):
    mapper = _tiny_mlp()
    path = tmp_path / "mlp.pt"
    mapper.save(path)
    metadata = {"mapper": {"kind": "mlp_full_head"}}
    meta_path = tmp_path / "mlp.pt.json"
    meta_path.write_text(json.dumps(metadata))

    loaded = load_runtime_mapper(path, meta_path)

    assert isinstance(loaded, GroupedHeadMLPMapper)
