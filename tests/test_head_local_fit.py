import torch

from verifier_anchored_sd.spec_decode.cache_state import CacheState, LayerKV
from verifier_anchored_sd.spec_decode.head_local_fit import fit_matched_head_mapper_from_cache_pairs


def _pair(seed: int, tokens: int = 16):
    g = torch.Generator().manual_seed(seed)
    source_layers = []
    for _ in range(2):
        k = torch.randn(1, 2, tokens, 2, generator=g)
        v = torch.randn(1, 2, tokens, 2, generator=g)
        source_layers.append(LayerKV(k, v))
    source = CacheState(source_layers)

    def project(kind: str):
        rows = []
        for h in range(2):
            parts = [
                (layer.key if kind == "k" else layer.value)[:, h]
                for layer in source.layers
            ]
            x = torch.cat(parts, dim=-1)
            w = torch.tensor([[1.0, -0.5], [0.3, 0.7], [-0.2, 0.4], [0.8, -0.1]])
            rows.append(x @ w + torch.tensor([0.1 * (h + 1), -0.2 * (h + 1)]))
        return torch.stack(rows, dim=1)

    draft = CacheState([LayerKV(project("k"), project("v"))])
    return source, draft


def test_streaming_matched_head_fit_recovers_independent_heads():
    train = [_pair(i) for i in range(8)]
    mapper = fit_matched_head_mapper_from_cache_pairs(
        lambda: iter(train),
        target_layers=2,
        draft_layers=1,
        kv_heads=2,
        head_dim=2,
        layer_selection=[[0, 1]],
        lambda_=1e-6,
        accumulation_device="cpu",
        layer_block_size=1,
        content_space=False,
    )
    assert mapper.metadata.head_mode == "matched"
    assert mapper.in_dim == 4

    source, expected = _pair(100)
    actual = mapper.map(source)
    assert torch.allclose(actual.layers[0].key, expected.layers[0].key, atol=2e-3, rtol=2e-3)
    assert torch.allclose(actual.layers[0].value, expected.layers[0].value, atol=2e-3, rtol=2e-3)
