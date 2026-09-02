import torch

from verifier_anchored_sd.spec_decode.cache_state import CacheState, LayerKV
from verifier_anchored_sd.spec_decode.target_to_draft_mapper import MapperMetadata, RidgeKVMapper


def test_mapper_can_mix_all_source_kv_heads():
    """A target draft head must be able to read features from every verifier KV head."""
    metadata = MapperMetadata(
        target_layers=1,
        draft_layers=1,
        target_kv_heads=2,
        draft_kv_heads=2,
        head_dim=2,
        layer_selection=[[0]],
        lambda_=0.01,
        content_space=False,
    )
    # Feature order for one selected layer is [h0:d0,d1, h1:d0,d1].
    weights = torch.zeros(1, 2, 2, 2, 4)
    bias = torch.zeros(1, 2, 2, 2)
    for kind in (0, 1):
        # draft head 0 copies verifier head 1; draft head 1 copies verifier head 0.
        weights[0, kind, 0, 0, 2] = 1.0
        weights[0, kind, 0, 1, 3] = 1.0
        weights[0, kind, 1, 0, 0] = 1.0
        weights[0, kind, 1, 1, 1] = 1.0

    mapper = RidgeKVMapper(metadata, weights, bias)
    key = torch.tensor([[[[1.0, 2.0]], [[10.0, 20.0]]]])
    value = key + 100.0
    mapped = mapper.map(CacheState([LayerKV(key, value)]))

    assert torch.equal(mapped.layers[0].key[0, 0, 0], torch.tensor([10.0, 20.0]))
    assert torch.equal(mapped.layers[0].key[0, 1, 0], torch.tensor([1.0, 2.0]))
    assert torch.equal(mapped.layers[0].value[0, 0, 0], torch.tensor([110.0, 120.0]))
    assert torch.equal(mapped.layers[0].value[0, 1, 0], torch.tensor([101.0, 102.0]))
