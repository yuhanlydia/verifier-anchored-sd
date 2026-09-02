import torch

from verifier_anchored_sd.spec_decode.cache_state import CacheState, LayerKV
from verifier_anchored_sd.spec_decode.target_to_draft_mapper import fit_ridge_mapper
from verifier_anchored_sd.spec_decode.verifier_cache_refresh import VerifierAnchoredCache


def mapper():
    obs = {(0, 0, kind): (torch.randn(20, 2), torch.randn(20, 2)) for kind in ("k", "v")}
    return fit_ridge_mapper(obs, target_layers=1, draft_layers=1, kv_heads=1, head_dim=2, layer_selection=[[0]])


def kv(n, value):
    x = torch.full((1, 1, n, 2), value)
    return CacheState([LayerKV(x.clone(), x.clone())])


def test_refresh_replaces_without_changing_length():
    state = VerifierAnchoredCache(kv(3, 1), mapper())
    state.append_pending(7, kv(1, 2))
    assert state.seq_len == 4
    expected = state.mapper.map(kv(1, 9))
    state.materialize_pending(7, kv(1, 9))
    assert state.seq_len == 4
    assert torch.allclose(state.draft_cache.layers[0].key[..., -1:, :], expected.layers[0].key)
