import pytest
import torch

from verifier_anchored_sd.spec_decode.cache_state import CacheState, LayerKV
from verifier_anchored_sd.spec_decode.target_to_draft_mapper import fit_ridge_mapper
from verifier_anchored_sd.spec_decode.verifier_cache_refresh import VerifierAnchoredCache


def test_pending_cannot_be_appended_twice():
    x = torch.randn(20, 2)
    m = fit_ridge_mapper({(0, 0, k): (x, x) for k in ("k", "v")}, target_layers=1, draft_layers=1, kv_heads=1, head_dim=2, layer_selection=[[0]])
    z = torch.zeros(1, 1, 2, 2)
    s = VerifierAnchoredCache(CacheState([LayerKV(z, z)]), m)
    one = CacheState([LayerKV(z[..., :1, :], z[..., :1, :])])
    s.append_pending(3, one)
    with pytest.raises(RuntimeError):
        s.append_pending(4, one)

