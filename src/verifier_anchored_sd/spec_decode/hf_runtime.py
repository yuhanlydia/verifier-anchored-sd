"""Optional Hugging Face runtime for the Qwen3 4B/1.7B pilot.

This module is imported lazily by benchmark scripts, so the core package and its
tests do not require Transformers.  It uses legacy tuple caches at the adapter
boundary; this is supported by current Transformers and avoids rebuilding a
DynamicCache object on every mapper call.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from .cache_state import CacheState, LayerKV
from .target_to_draft_mapper import RidgeKVMapper
from .verifier_cache_refresh import VerifierAnchoredCache


def _concat_steps(steps: list[CacheState]) -> CacheState:
    if not steps:
        raise ValueError("cannot concatenate an empty KV step list")
    result = steps[0].clone()
    for step in steps[1:]:
        result.append(step)
    return result


def cache_state_from_hf(past) -> CacheState:
    if hasattr(past, "key_cache") and hasattr(past, "value_cache"):
        pairs = zip(past.key_cache, past.value_cache)
    else:
        pairs = past
    return CacheState(LayerKV(k, v) for k, v in pairs)


@dataclass
class Forward:
    logits: torch.Tensor
    cache: CacheState


def forward_incremental(model, input_ids: torch.Tensor, cache: CacheState | None = None) -> Forward:
    """Run a real incremental forward and return only the new-token KV."""
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, sequence]")
    past_len = 0 if cache is None else cache.seq_len
    device = input_ids.device
    positions = torch.arange(past_len, past_len + input_ids.shape[1], device=device).unsqueeze(0)
    attention_mask = torch.ones(
        (input_ids.shape[0], past_len + input_ids.shape[1]), device=device, dtype=torch.long
    )
    with torch.inference_mode():
        kwargs = {
            "input_ids": input_ids,
            "past_key_values": None if cache is None else cache.as_tuple(),
            "attention_mask": attention_mask,
            "position_ids": positions,
            "use_cache": True,
            "return_dict": True,
        }
        try:
            out = model(**kwargs, cache_position=positions[0])
        except TypeError as exc:
            if "cache_position" not in str(exc):
                raise
            out = model(**kwargs)
    full = cache_state_from_hf(out.past_key_values)
    new = full.slice(past_len)
    return Forward(out.logits, new)


@dataclass
class ProposalBlock:
    token_ids: list[int]
    draft_probs: torch.Tensor
    target_probs: torch.Tensor
    target_kv: CacheState
    draft_kv: CacheState
    next_target_probs: torch.Tensor
    next_draft_probs: torch.Tensor


class QwenPairRuntime:
    """Incremental Qwen pair runtime implementing initial bridge + refresh.

    ``target_next_probs`` and ``draft_next_probs`` are the distributions before
    the next proposed token.  Keeping them explicitly avoids the common off-by-one
    error where verification logits are incorrectly aligned to proposal tokens.
    """

    def __init__(self, target, draft, mapper: RidgeKVMapper, *, temperature: float = 1.0, seed: int = 0,
                 init_mode: str = "mapped", refresh: bool = True):
        self.target, self.draft, self.mapper = target, draft, mapper
        self.temperature = temperature
        if init_mode not in {"mapped", "native"}:
            raise ValueError("init_mode must be mapped or native")
        self.init_mode, self.refresh = init_mode, refresh
        self.generator = torch.Generator(device=next(draft.parameters()).device).manual_seed(seed)
        self.target_cache: CacheState | None = None
        self.anchored: VerifierAnchoredCache | None = None
        self.target_next_probs: torch.Tensor | None = None
        self.draft_next_probs: torch.Tensor | None = None
        self.prompt_last: int | None = None
        self.accepted_lengths: list[int] = []
        self.block_emitted_lengths: list[int] = []

    @staticmethod
    def _probs(logits: torch.Tensor, temperature: float) -> torch.Tensor:
        return torch.softmax(logits[:, -1, :].float() / temperature, dim=-1)

    def initialize(self, prompt_ids: Sequence[int]) -> None:
        self.accepted_lengths = []
        self.block_emitted_lengths = []
        ids = torch.tensor([list(prompt_ids)], device=next(self.target.parameters()).device)
        if ids.shape[1] < 1:
            raise ValueError("prompt must contain at least one token")
        target_full = forward_incremental(self.target, ids)
        # forward_incremental returns only new KV when cache is None: the full prompt.
        self.target_cache = target_full.cache
        self.anchored = VerifierAnchoredCache(self.target_cache, self.mapper)
        self.target_next_probs = self._probs(target_full.logits, self.temperature)
        self.prompt_last = int(ids[0, -1])
        if self.init_mode == "native":
            native_full = forward_incremental(self.draft, ids)
            self.anchored.draft_cache = native_full.cache
            self.draft_next_probs = self._probs(native_full.logits, self.temperature)
            return
        # Query the draft head at the prompt boundary while discarding the native
        # K/V for the last prompt token. Persistent draft state remains mapped.
        draft_prefix = self.anchored.draft_cache.slice(0, self.anchored.seq_len - 1).clone()
        query = forward_incremental(self.draft, ids[:, -1:], draft_prefix)
        self.draft_next_probs = self._probs(query.logits, self.temperature)

    def _sample(self, probs: torch.Tensor) -> int:
        return int(torch.multinomial(probs[0], 1, generator=self.generator).item())

    def _materialize_pending(self) -> None:
        assert self.target_cache is not None and self.anchored is not None
        if self.anchored.pending is None:
            return
        token = self.anchored.pending.token_id
        ids = torch.tensor([[token]], device=self.target_cache.layers[0].key.device)
        target_step = forward_incremental(self.target, ids, self.target_cache)
        self.target_cache.append(target_step.cache)
        if self.refresh:
            self.anchored.materialize_pending(token, target_step.cache)
        else:
            self.anchored.pending = None
        self.target_next_probs = self._probs(target_step.logits, self.temperature)
        # Re-query draft using the now-anchored correction KV; discard new KV.
        prefix = self.anchored.draft_cache.slice(0, self.anchored.seq_len - 1).clone()
        draft_step = forward_incremental(self.draft, ids, prefix)
        self.draft_next_probs = self._probs(draft_step.logits, self.temperature)

    def propose(self, gamma: int) -> ProposalBlock:
        if self.target_cache is None or self.anchored is None:
            raise RuntimeError("call initialize first")
        self._materialize_pending()
        assert self.target_next_probs is not None and self.draft_next_probs is not None
        q_rows, tokens, temp = [], [], []
        draft_cache = self.anchored.draft_cache.clone()
        q = self.draft_next_probs
        for _ in range(gamma):
            q_rows.append(q[0])
            token = self._sample(q)
            tokens.append(token)
            ids = torch.tensor([[token]], device=q.device)
            step = forward_incremental(self.draft, ids, draft_cache)
            temp.append(step.cache)
            draft_cache.append(step.cache)
            q = self._probs(step.logits, self.temperature)
        ids = torch.tensor([tokens], device=self.target_next_probs.device)
        verify = forward_incremental(self.target, ids, self.target_cache)
        # p(y_1) comes from the previous target boundary; p(y_i), i>1, is the
        # logit after y_{i-1}. The final verify logit is retained for all-accepted
        # next-round use by the caller.
        p_rows = [self.target_next_probs[0]]
        if gamma > 1:
            p_rows.extend(torch.softmax(verify.logits[0, :-1].float() / self.temperature, dim=-1))
        return ProposalBlock(
            tokens, torch.stack(q_rows), torch.stack(p_rows), verify.cache,
            _concat_steps(temp),
            self._probs(verify.logits, self.temperature), q,
        )

    def commit(self, accepted: int, correction: int | None, proposal: ProposalBlock) -> None:
        assert self.target_cache is not None and self.anchored is not None
        accepted_target = proposal.target_kv.slice(0, accepted)
        self.target_cache.append(accepted_target)
        if self.refresh:
            self.anchored.append_verified(accepted_target)
        else:
            self.anchored.draft_cache.append(proposal.draft_kv.slice(0, accepted))
        if correction is None:
            if accepted != len(proposal.token_ids):
                raise ValueError("correction is required after a partial block")
            self.target_next_probs = proposal.next_target_probs
            self.draft_next_probs = proposal.next_draft_probs
            return
        ids = torch.tensor([[correction]], device=self.target_cache.layers[0].key.device)
        native = forward_incremental(self.draft, ids, self.anchored.draft_cache).cache
        self.anchored.append_pending(correction, native)

    def generate(self, prompt_ids: Sequence[int], max_new_tokens: int, gamma: int = 4) -> list[int]:
        self.initialize(prompt_ids)
        output: list[int] = []
        while len(output) < max_new_tokens:
            proposal = self.propose(gamma)
            from .exact_sd import exact_spec_accept

            result = exact_spec_accept(
                proposal.token_ids, proposal.draft_probs, proposal.target_probs,
                generator=self.generator,
            )
            self.accepted_lengths.append(result.accepted_length)
            self.block_emitted_lengths.append(result.accepted_length + (1 if result.correction is not None else 0))
            self.commit(result.accepted_length, result.correction, proposal)
            output.extend(result.accepted)
            if result.correction is not None:
                output.append(result.correction)
        return output[:max_new_tokens]
