import torch

from verifier_anchored_sd.spec_decode.cache_state import CacheState, LayerKV, RotaryFactors
from verifier_anchored_sd.spec_decode.target_to_draft_mapper import MapperMetadata, RidgeKVMapper


def _factors(angles: torch.Tensor) -> RotaryFactors:
    freqs = torch.cat((angles, angles), dim=-1)
    return RotaryFactors(torch.cos(freqs), torch.sin(freqs))


def test_content_space_mapper_strips_source_rope_and_applies_draft_rope():
    source_rotary = _factors(torch.tensor([[0.1, 0.2], [0.3, 0.4]]))
    draft_rotary = _factors(torch.tensor([[0.5, 0.6], [0.7, 0.8]]))
    content_key = torch.tensor(
        [[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]]]
    )
    source_key = source_rotary.apply(content_key)
    value = torch.tensor(
        [[[[2.0, 4.0, 6.0, 8.0], [1.0, 3.0, 5.0, 7.0]]]]
    )
    source = CacheState([LayerKV(source_key, value)], rotary=source_rotary)

    metadata = MapperMetadata(
        target_layers=1,
        draft_layers=1,
        target_kv_heads=1,
        draft_kv_heads=1,
        head_dim=4,
        layer_selection=[[0]],
        lambda_=0.01,
        content_space=True,
    )
    weights = torch.zeros(1, 2, 1, 4, 4)
    weights[0, 0, 0] = torch.eye(4)
    weights[0, 1, 0] = torch.eye(4)
    mapper = RidgeKVMapper(metadata, weights, torch.zeros(1, 2, 1, 4))

    mapped = mapper.map(source, draft_rotary=draft_rotary)

    assert torch.allclose(mapped.layers[0].key, draft_rotary.apply(content_key), atol=1e-5)
    assert torch.allclose(mapped.layers[0].value, value, atol=1e-5)
    assert mapped.rotary is draft_rotary


def test_rotary_inverse_round_trip():
    rotary = _factors(torch.tensor([[0.2, 0.4], [0.6, 0.8]]))
    x = torch.randn(1, 2, 2, 4)
    assert torch.allclose(rotary.apply(rotary.apply(x), inverse=True), x, atol=1e-5)
