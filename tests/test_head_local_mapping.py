import torch

from verifier_anchored_sd.spec_decode.cache_state import CacheState, LayerKV
from verifier_anchored_sd.spec_decode.target_to_draft_mapper import MapperMetadata, RidgeKVMapper


def test_head_local_mapper_reads_only_matching_source_head():
    metadata = MapperMetadata(
        target_layers=2,
        draft_layers=1,
        target_kv_heads=2,
        draft_kv_heads=2,
        head_dim=2,
        layer_selection=[[0, 1]],
        lambda_=0.01,
        content_space=False,
        head_mode="matched",
    )
    # matched-head width = selected_layers * head_dim = 4, not * source_heads (=8)
    weights = torch.zeros(1, 2, 2, 2, 4)
    bias = torch.zeros(1, 2, 2, 2)
    # Both output heads copy the first component from each of their own two source layers.
    weights[0, :, :, 0, 0] = 1.0
    weights[0, :, :, 1, 2] = 1.0
    mapper = RidgeKVMapper(metadata, weights, bias)

    l0 = torch.zeros(1, 2, 1, 2)
    l1 = torch.zeros(1, 2, 1, 2)
    l0[0, 0, 0] = torch.tensor([1.0, 2.0])
    l1[0, 0, 0] = torch.tensor([3.0, 4.0])
    l0[0, 1, 0] = torch.tensor([10.0, 20.0])
    l1[0, 1, 0] = torch.tensor([30.0, 40.0])
    cache = CacheState([LayerKV(l0, l0.clone()), LayerKV(l1, l1.clone())])

    mapped = mapper.map(cache)
    assert mapped.layers[0].key[0, 0, 0].tolist() == [1.0, 3.0]
    assert mapped.layers[0].key[0, 1, 0].tolist() == [10.0, 30.0]

    # Perturb source head 1 only: target/draft head 0 must be unchanged.
    perturbed = cache.clone()
    perturbed.layers[0].key[:, 1].add_(1000)
    perturbed.layers[1].key[:, 1].add_(1000)
    remapped = mapper.map(perturbed)
    assert torch.equal(remapped.layers[0].key[:, 0], mapped.layers[0].key[:, 0])
    assert not torch.equal(remapped.layers[0].key[:, 1], mapped.layers[0].key[:, 1])
