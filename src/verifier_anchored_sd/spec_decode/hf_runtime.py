"""Incremental Hugging Face runtime for the Qwen3 4B/1.7B pilot.

The runtime keeps the verifier exact.  Cross-model mapping is used only to build
or refresh the draft cache.  RoPE factors are captured from each model's own
rotary module so the affine bridge is applied in position-free content space.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass

import torch

from .cache_state import CacheState, LayerKV, RotaryFactors
from .target_to_draft_mapper import RidgeKVMapper
from .verifier_cache_refresh import VerifierAnchoredCache


def _concat_steps(steps: list[CacheState]) -> CacheState:
    if not steps:
        raise ValueError("cannot concatenate an empty KV step list")
    result = steps[0].clone()
    for step in steps[1:]:
        result.append(step)
    return result


def _legacy_layers(past) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if hasattr(past, "to_legacy_cache"):
        past = past.to_legacy_cache()
    if isinstance(past, (tuple, list)):
        return [(layer[0], layer[1]) for layer in past]
    if hasattr(past, "key_cache") and hasattr(past, "value_cache"):
        return list(zip(past.key_cache, past.value_cache))
    if hasattr(past, "layers"):
        return [(layer.keys, layer.values) for layer in past.layers]
    raise RuntimeError("unsupported Transformers cache representation")


def cache_state_from_hf(past) -> CacheState:
    return CacheState(LayerKV(k, v) for k, v in _legacy_layers(past))


def _rotary_module(model):
    candidates = [
        getattr(getattr(model, "model", None), "rotary_emb", None),
        getattr(getattr(getattr(model, "model", None), "model", None), "rotary_emb", None),
        getattr(model, "rotary_emb", None),
    ]
    for candidate in candidates:
        if candidate is not None:
            return candidate
    raise RuntimeError("model does not expose a supported model-level rotary embedding")


@torch.inference_mode()
def capture_rotary_factors(model, position_ids: torch.Tensor) -> RotaryFactors:
    """Capture the exact cos/sin factors emitted by the receiver/source model."""
    rotary = _rotary_module(model)
    embeddings = model.get_input_embeddings().weight
    positions = position_ids.to(embeddings.device)
    dummy = torch.empty((*positions.shape, 1), device=embeddings.device, dtype=embeddings.dtype)
    output = rotary(dummy, positions)
    if not isinstance(output, tuple) or len(output) < 2:
        raise RuntimeError("rotary module did not return (cos, sin)")
    return RotaryFactors(output[0], output[1], interleaved=False)


@dataclass
class Forward:
    logits: torch.Tensor
    cache: CacheState


def forward_incremental(
    model,
    input_ids: torch.Tensor,
    cache: CacheState | None = None,
    *,
    inference: bool = True,
    capture_rotary: bool = True,
) -> Forward:
    """Run a real incremental forward and return only the newly materialized KV."""
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, sequence]")
    past_len = 0 if cache is None else cache.seq_len
    device = input_ids.device
    positions = torch.arange(past_len, past_len + input_ids.shape[1], device=device).unsqueeze(0)
    if input_ids.shape[0] != 1:
        positions = positions.expand(input_ids.shape[0], -1)
    attention_mask = torch.ones(
        (input_ids.shape[0], past_len + input_ids.shape[1]), device=device, dtype=torch.long
    )
    context = torch.inference_mode() if inference else nullcontext()
    with context:
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
    layers = _legacy_layers(out.past_key_values)
    new_layers = [LayerKV(k[..., past_len:, :], v[..., past_len:, :]) for k, v in layers]
    rotary = capture_rotary_factors(model, positions) if capture_rotary else None
    return Forward(out.logits, CacheState(new_layers, rotary=rotary))


@dataclass
class ProposalBlock:
    token_ids: list[int]
    draft_probs: torch.Tensor
    target_probs: torch.Tensor
    target_kv: CacheState
    draft_kv: CacheState
    draft_rotary: RotaryFactors
    next_target_probs: torch.Tensor
    next_draft_probs: torch.Tensor


class QwenPairRuntime:
    """Exact speculative decoding with initial bridge and optional verifier refresh."""

    def __init__(
        self,
        target,
        draft,
        mapper: RidgeKVMapper,
        *,
        temperature: float = 1.0,
        seed: int = 0,
        init_mode: str = "mapped",
        refresh: bool = True,
    ):
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
        self.accepted_lengths: list[int] = []
        self.block_emitted_lengths: list[int] = []

    @staticmethod
    def _probs(logits: torch.Tensor, temperature: float) -> torch.Tensor:
        return torch.softmax(logits[:, -1, :].float() / temperature, dim=-1)

    def _draft_rotary(self, start: int, length: int) -> RotaryFactors:
        device = next(self.draft.parameters()).device
        positions = torch.arange(start, start + length, device=device).unsqueeze(0)
        return capture_rotary_factors(self.draft, positions)

    def initialize(self, prompt_ids: Sequence[int]) -> None:
        self.accepted_lengths = []
        self.block_emitted_lengths = []
        ids = torch.tensor([list(prompt_ids)], device=next(self.target.parameters()).device)
        if ids.shape[1] < 1:
            raise ValueError("prompt must contain at least one token")
        target_full = forward_incremental(self.target, ids)
        self.target_cache = target_full.cache
        self.target_next_probs = self._probs(target_full.logits, self.temperature)
        if self.init_mode == "native":
            native_full = forward_incremental(self.draft, ids.to(next(self.draft.parameters()).device))
            self.anchored = VerifierAnchoredCache.from_native(native_full.cache, self.mapper)
            self.draft_next_probs = self._probs(native_full.logits, self.temperature)
            return

        draft_rotary = self._draft_rotary(0, ids.shape[1])
        self.anchored = VerifierAnchoredCache(self.target_cache, self.mapper, draft_rotary)
        # Obtain the boundary distribution without replaying the prompt.  As in
        # cross-model handoff, only the final prompt token is processed by the draft.
        draft_prefix = self.anchored.draft_cache.slice(0, self.anchored.seq_len - 1).clone()
        query_ids = ids[:, -1:].to(next(self.draft.parameters()).device)
        query = forward_incremental(self.draft, query_ids, draft_prefix)
        self.draft_next_probs = self._probs(query.logits, self.temperature)

    def _sample(self, probs: torch.Tensor) -> int:
        return int(torch.multinomial(probs[0], 1, generator=self.generator).item())

    def _materialize_pending(self) -> None:
        assert self.target_cache is not None and self.anchored is not None
        if self.anchored.pending is None:
            return
        token = self.anchored.pending.token_id
        position = self.target_cache.seq_len
        ids = torch.tensor([[token]], device=self.target_cache.layers[0].key.device)
        target_step = forward_incremental(self.target, ids, self.target_cache)
        self.target_cache.append(target_step.cache)
        if self.refresh:
            self.anchored.materialize_pending(token, target_step.cache, self._draft_rotary(position, 1))
        else:
            self.anchored.pending = None
        self.target_next_probs = self._probs(target_step.logits, self.temperature)
        # Re-query the boundary distribution after refresh; the query's new KV is discarded.
        prefix = self.anchored.draft_cache.slice(0, self.anchored.seq_len - 1).clone()
        draft_ids = ids.to(next(self.draft.parameters()).device)
        draft_step = forward_incremental(self.draft, draft_ids, prefix)
        self.draft_next_probs = self._probs(draft_step.logits, self.temperature)

    def propose(self, gamma: int) -> ProposalBlock:
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        if self.target_cache is None or self.anchored is None:
            raise RuntimeError("call initialize first")
        self._materialize_pending()
        assert self.target_next_probs is not None and self.draft_next_probs is not None
        start = self.target_cache.seq_len
        q_rows, tokens, temp = [], [], []
        draft_cache = self.anchored.draft_cache.clone()
        q = self.draft_next_probs
        for _ in range(gamma):
            q_rows.append(q[0])
            token = self._sample(q)
            tokens.append(token)
            ids = torch.tensor([[token]], device=next(self.draft.parameters()).device)
            step = forward_incremental(self.draft, ids, draft_cache)
            temp.append(step.cache)
            draft_cache.append(step.cache)
            q = self._probs(step.logits, self.temperature)
        verify_ids = torch.tensor([tokens], device=next(self.target.parameters()).device)
        verify = forward_incremental(self.target, verify_ids, self.target_cache)
        p_rows = [self.target_next_probs[0]]
        if gamma > 1:
            p_rows.extend(torch.softmax(verify.logits[0, :-1].float() / self.temperature, dim=-1))
        return ProposalBlock(
            token_ids=tokens,
            draft_probs=torch.stack(q_rows),
            target_probs=torch.stack(p_rows),
            target_kv=verify.cache,
            draft_kv=_concat_steps(temp),
            draft_rotary=self._draft_rotary(start, gamma),
            next_target_probs=self._probs(verify.logits, self.temperature),
            next_draft_probs=q,
        )

    def commit(self, accepted: int, correction: int | None, proposal: ProposalBlock) -> None:
        assert self.target_cache is not None and self.anchored is not None
        if not 0 <= accepted <= len(proposal.token_ids):
            raise ValueError("accepted length is outside proposal block")
        if accepted:
            accepted_target = proposal.target_kv.slice(0, accepted)
            self.target_cache.append(accepted_target)
            if self.refresh:
                self.anchored.append_verified(
                    accepted_target, proposal.draft_rotary.slice(0, accepted)
                )
            else:
                self.anchored.draft_cache.append(proposal.draft_kv.slice(0, accepted))
        if correction is None:
            if accepted != len(proposal.token_ids):
                raise ValueError("correction is required after a partial block")
            self.target_next_probs = proposal.next_target_probs
            if self.refresh:
                last_token = proposal.token_ids[-1]
                ids = torch.tensor(
                    [[last_token]], device=next(self.draft.parameters()).device
                )
                prefix = self.anchored.draft_cache.slice(0, self.anchored.seq_len - 1).clone()
                refreshed = forward_incremental(self.draft, ids, prefix)
                self.draft_next_probs = self._probs(refreshed.logits, self.temperature)
            else:
                self.draft_next_probs = proposal.next_draft_probs
            return
        ids = torch.tensor([[correction]], device=next(self.draft.parameters()).device)
        native = forward_incremental(self.draft, ids, self.anchored.draft_cache).cache
        self.anchored.append_pending(correction, native)

    def generate(self, prompt_ids: Sequence[int], max_new_tokens: int, gamma: int = 4) -> list[int]:
        self.initialize(prompt_ids)
        output: list[int] = []
        while len(output) < max_new_tokens:
            proposal = self.propose(gamma)
            from .exact_sd import exact_spec_accept

            result = exact_spec_accept(
                proposal.token_ids,
                proposal.draft_probs,
                proposal.target_probs,
                generator=self.generator,
            )
            self.accepted_lengths.append(result.accepted_length)
            emitted = result.accepted_length + (1 if result.correction is not None else 0)
            self.block_emitted_lengths.append(emitted)
            self.commit(result.accepted_length, result.correction, proposal)
            output.extend(result.accepted)
            if result.correction is not None:
                output.append(result.correction)
        return output[:max_new_tokens]
