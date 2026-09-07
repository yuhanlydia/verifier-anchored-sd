import torch

from verifier_anchored_sd.spec_decode.cache_state import CacheState, LayerKV
from verifier_anchored_sd.spec_decode.head_local_fit import select_source_layers_by_r2


def synthetic_pair(seed: int):
    generator = torch.Generator().manual_seed(seed)
    source_layers = []
    for _ in range(3):
        key = torch.randn(1, 2, 12, 2, generator=generator)
        value = torch.randn(1, 2, 12, 2, generator=generator)
        source_layers.append(LayerKV(key, value))
    weight = torch.tensor([[1.5, -0.2], [0.3, 0.8]])
    key = source_layers[1].key @ weight
    value = source_layers[1].value @ weight
    return CacheState(source_layers), CacheState([LayerKV(key, value)])


def test_r2_selection_finds_constructed_source_layer():
    pairs = [synthetic_pair(seed) for seed in range(4)]

    selected, scores = select_source_layers_by_r2(
        lambda: iter(pairs),
        target_layers=3,
        draft_layers=1,
        kv_heads=2,
        head_dim=2,
        top_k=1,
        device="cpu",
        layer_block_size=1,
        content_space=False,
    )

    assert selected == [[1]]
    assert scores[0][1] > 0.999
    assert scores[0][1] > scores[0][0]


def test_r2_selection_uses_source_index_as_stable_tie_breaker():
    source, draft = synthetic_pair(1)
    duplicate = CacheState([source.layers[1], source.layers[1]])

    selected, _ = select_source_layers_by_r2(
        [(duplicate, draft)],
        target_layers=2,
        draft_layers=1,
        kv_heads=2,
        head_dim=2,
        top_k=2,
        device="cpu",
        layer_block_size=1,
        content_space=False,
    )

    assert selected == [[0, 1]]
