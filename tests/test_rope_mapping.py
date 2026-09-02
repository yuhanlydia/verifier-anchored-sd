import torch

from verifier_anchored_sd.spec_decode.cache_state import CacheState, LayerKV
from verifier_anchored_sd.spec_decode.target_to_draft_mapper import fit_ridge_mapper


def test_mapper_maps_layer_head_shapes():
    observations = {}
    for kind in ("k", "v"):
        observations[(0, 0, kind)] = (torch.randn(32, 8), torch.randn(32, 4))
        observations[(0, 1, kind)] = (torch.randn(32, 8), torch.randn(32, 4))
    mapper = fit_ridge_mapper(observations, target_layers=2, draft_layers=1, kv_heads=2, head_dim=4,
                              layer_selection=[[0, 1]], lambda_=0.01)
    target = CacheState([LayerKV(torch.randn(1, 2, 5, 4), torch.randn(1, 2, 5, 4)) for _ in range(2)])
    mapped = mapper.map(target)
    assert mapped.layers[0].key.shape == (1, 2, 5, 4)

