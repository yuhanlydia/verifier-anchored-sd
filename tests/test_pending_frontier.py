import pytest
import torch

from verifier_anchored_sd.spec_decode.cache_state import CacheState, LayerKV
from verifier_anchored_sd.spec_decode.target_to_draft_mapper import fit_ridge_mapper
from verifier_anchored_sd.spec_decode.verifier_cache_refresh import VerifierAnchoredCache


def test_pending_token_id_is_checked():
    x = torch.randn(20, 2)
    m = fit_ridge_mapper({(0, 0, k): (x, x) for k in ("k", "v")}, target_layers=1, draft_layers=1, kv_heads=1, head_dim=2, layer_selection=[[0]])
    z = torch.zeros(1, 1, 1, 2)
    s = VerifierAnchoredCache(CacheState([LayerKV(z, z)]), m)
    s.append_pending(9, CacheState([LayerKV(z, z)]))
    with pytest.raises(ValueError):
        s.materialize_pending(8, CacheState([LayerKV(z, z)]))

