import torch

from verifier_anchored_sd.spec_decode.cache_state import CacheState, LayerKV
from verifier_anchored_sd.spec_decode.hf_runtime import Forward, complete_incremental_forward


def _cache(tokens: int, value: float) -> CacheState:
    tensor = torch.full((1, 1, tokens, 2), value)
    return CacheState([LayerKV(tensor.clone(), tensor.clone())])


def test_complete_incremental_forward_keeps_history_and_frontier():
    history = _cache(3, 1.0)
    step = Forward(torch.tensor([[[2.0, 3.0]]]), _cache(1, 2.0))

    complete = complete_incremental_forward(history, step)

    assert complete.cache.seq_len == 4
    assert history.seq_len == 3
    assert complete.logits is step.logits
