import torch

from verifier_anchored_sd.spec_decode.cache_state import CacheState, LayerKV, RotaryFactors
from verifier_anchored_sd.spec_decode.hf_runtime import (
    Forward,
    QwenPairRuntime,
    _model_input_device,
)
from verifier_anchored_sd.spec_decode.verifier_cache_refresh import VerifierAnchoredCache


def kv(tokens: int, value: float) -> CacheState:
    tensor = torch.full((1, 1, tokens, 2), value)
    return CacheState([LayerKV(tensor.clone(), tensor.clone())])


class IdentityMapper:
    def __init__(self, mapped_value: float = 5.0):
        self.mapped_value = mapped_value
        self.map_lengths = []

    def map(self, cache, *, draft_rotary=None):
        self.map_lengths.append(cache.seq_len)
        return kv(cache.seq_len, self.mapped_value)


class FakeModel(torch.nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def get_input_embeddings(self):
        return self


def test_accepted_only_resolves_pending_without_replacing_native_kv():
    state = VerifierAnchoredCache(kv(2, 1), IdentityMapper())
    next_probs = torch.tensor([[0.2, 0.8]])
    state.append_pending(7, kv(1, 3), next_probs=next_probs)
    before = state.draft_cache.layers[0].key[..., -1:, :].clone()

    actual_probs = state.resolve_pending_native(7)

    assert torch.equal(state.draft_cache.layers[0].key[..., -1:, :], before)
    assert torch.equal(actual_probs, next_probs)
    assert state.pending is None


def test_mapped_native_frontier_initialization_keeps_kv_from_logits_forward(monkeypatch):
    target = FakeModel("target")
    draft = FakeModel("draft")
    mapper = IdentityMapper(mapped_value=5)

    def fake_forward(model, input_ids, cache=None, **_kwargs):
        if model.name == "target":
            return Forward(torch.tensor([[[2.0, 0.0]]]), kv(input_ids.shape[1], 10))
        if cache is None:
            return Forward(torch.tensor([[[0.0, 2.0]]]), kv(input_ids.shape[1], 1))
        assert cache.seq_len == 2
        return Forward(torch.tensor([[[0.0, 3.0]]]), kv(1, 7))

    def fake_rotary(_model, positions):
        shape = (1, positions.shape[1], 2)
        return RotaryFactors(torch.ones(shape), torch.zeros(shape))

    monkeypatch.setattr(
        "verifier_anchored_sd.spec_decode.hf_runtime.forward_incremental", fake_forward
    )
    monkeypatch.setattr(
        "verifier_anchored_sd.spec_decode.hf_runtime.capture_rotary_factors", fake_rotary
    )
    runtime = QwenPairRuntime(
        target,
        draft,
        mapper,
        init_mode="mapped_native_frontier",
        refresh_policy="accepted_only",
    )

    runtime.initialize([1, 2, 3])

    assert mapper.map_lengths == [2]
    assert runtime.anchored.seq_len == 3
    assert torch.equal(
        runtime.anchored.draft_cache.layers[0].key[..., :2, :],
        kv(2, 5).layers[0].key,
    )
    assert torch.equal(
        runtime.anchored.draft_cache.layers[0].key[..., -1:, :],
        kv(1, 7).layers[0].key,
    )
    assert torch.allclose(runtime.draft_next_probs, torch.softmax(torch.tensor([[0.0, 3.0]]), -1))


def test_runtime_uses_embedding_device_for_accelerate_offload_models():
    class Offloaded(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.offloaded_first = torch.nn.Parameter(torch.empty(1, device="meta"))
            self.embedding = torch.nn.Embedding(4, 2)

        def get_input_embeddings(self):
            return self.embedding

    assert _model_input_device(Offloaded()) == torch.device("cpu")
