import torch

from verifier_anchored_sd.spec_decode.cache_state import CacheState, LayerKV


def cache(n, value):
    x = torch.full((1, 2, n, 4), value)
    return CacheState([LayerKV(x.clone(), x.clone())])


def test_append_and_slice_preserve_positions():
    state = cache(3, 1)
    state.append(cache(2, 2))
    assert state.seq_len == 5
    assert state.slice(3, 5).layers[0].key.equal(cache(2, 2).layers[0].key)

