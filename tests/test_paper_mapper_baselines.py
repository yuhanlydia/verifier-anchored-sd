import torch

from verifier_anchored_sd.paper_mapper_baselines import (
    GroupedHeadMLPMapper,
    MLPMapperMetadata,
    cache_state_to_kvbridge,
    model_signature_from_manifest,
)
from verifier_anchored_sd.spec_decode.cache_state import CacheState, LayerKV, RotaryFactors


def _rotary(tokens: int, dim: int) -> RotaryFactors:
    cos = torch.ones(1, tokens, dim)
    sin = torch.zeros(1, tokens, dim)
    return RotaryFactors(cos, sin)


def test_cache_state_to_kvbridge_preserves_geometry_and_rotary():
    rotary = _rotary(3, 4)
    cache = CacheState(
        [LayerKV(torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4)) for _ in range(2)],
        rotary=rotary,
    )

    converted = cache_state_to_kvbridge(cache)

    assert converted.shape == (2, 1, 2, 3, 4)
    assert converted.rotary is not None
    assert torch.equal(converted.rotary.cos, rotary.cos)
    assert torch.equal(converted.rotary.sin, rotary.sin)


def test_model_signature_from_manifest_uses_exact_identity():
    manifest = {
        "model": {
            "id": "Qwen/Qwen3-8B",
            "revision": "abc123",
            "tokenizer_hash": "tok",
            "layers": 36,
            "kv_heads": 8,
            "head_dim": 128,
            "architecture": "Qwen3ForCausalLM",
        }
    }

    signature = model_signature_from_manifest(manifest)

    assert signature.model_id == "Qwen/Qwen3-8B"
    assert signature.revision == "abc123"
    assert signature.num_layers == 36
    assert signature.num_kv_heads == 8
    assert signature.head_dim == 128
    assert signature.tokenizer_hash == "tok"


def _identity_like_mlp() -> GroupedHeadMLPMapper:
    metadata = MLPMapperMetadata(
        target_layers=1,
        draft_layers=1,
        target_kv_heads=2,
        draft_kv_heads=2,
        head_dim=2,
        layer_selection=[[0]],
        content_space=True,
        hidden_dim=2,
    )
    # Full-head feature order for one source layer is head-major => 4 input dims.
    # Head 0 reads dims 0:2, head 1 reads dims 2:4. ReLU is exact for positive test inputs.
    w1 = torch.zeros(1, 2, 2, 4, 2)
    w2 = torch.zeros(1, 2, 2, 2, 2)
    w3 = torch.zeros(1, 2, 2, 2, 2)
    b1 = torch.zeros(1, 2, 2, 2)
    b2 = torch.zeros(1, 2, 2, 2)
    b3 = torch.zeros(1, 2, 2, 2)
    for kind in range(2):
        w1[0, kind, 0, 0, 0] = 1
        w1[0, kind, 0, 1, 1] = 1
        w1[0, kind, 1, 2, 0] = 1
        w1[0, kind, 1, 3, 1] = 1
        w2[0, kind, 0] = torch.eye(2)
        w2[0, kind, 1] = torch.eye(2)
        w3[0, kind, 0] = torch.eye(2)
        w3[0, kind, 1] = torch.eye(2)
    return GroupedHeadMLPMapper(metadata, w1, b1, w2, b2, w3, b3)


def test_grouped_mlp_uses_cross_head_features_but_independent_target_heads():
    mapper = _identity_like_mlp()
    rotary = _rotary(2, 2)
    key = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]])
    value = key + 10.0
    source = CacheState([LayerKV(rotary.apply(key), value)], rotary=rotary)

    mapped = mapper.map(source, draft_rotary=rotary).to_content_space()

    assert torch.allclose(mapped.layers[0].key, key)
    assert torch.allclose(mapped.layers[0].value, value)


def test_grouped_mlp_save_load_round_trip(tmp_path):
    mapper = _identity_like_mlp()
    path = tmp_path / "mlp.pt"
    mapper.save(path)
    loaded = GroupedHeadMLPMapper.load(path)

    assert loaded.metadata.hidden_dim == 2
    assert torch.equal(loaded.w1, mapper.w1)
    assert torch.equal(loaded.w3, mapper.w3)
