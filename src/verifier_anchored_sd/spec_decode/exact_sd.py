"""Model-agnostic exact speculative rejection sampling."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass
class SpeculativeResult:
    accepted: list[int]
    correction: int | None
    rejected_at: int | None

    @property
    def accepted_length(self) -> int:
        return len(self.accepted)


def _normalized(x: torch.Tensor) -> torch.Tensor:
    x = x.float().clamp_min(0)
    total = x.sum()
    if not torch.isfinite(total) or total <= 0:
        raise ValueError("probability vector has no mass")
    return x / total


def exact_spec_accept(
    proposals: Sequence[int],
    draft_probs: torch.Tensor,
    target_probs: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> SpeculativeResult:
    """Accept a proposal block while preserving the exact target distribution.

    ``draft_probs`` and ``target_probs`` are ``[gamma, vocab]`` distributions and
    correspond to the conditional state at each proposal position. At the first
    rejection, the correction is sampled from ``normalize(max(p-q, 0))``.
    """
    if draft_probs.ndim != 2 or target_probs.shape != draft_probs.shape:
        raise ValueError("probabilities must both have shape [gamma, vocab]")
    if len(proposals) != draft_probs.shape[0]:
        raise ValueError("proposal count does not match probability rows")
    accepted: list[int] = []
    for i, token in enumerate(proposals):
        p, q = _normalized(target_probs[i]), _normalized(draft_probs[i])
        if not 0 <= token < p.numel():
            raise ValueError(f"proposal token {token} outside vocabulary")
        q_token = q[token]
        if q_token > 0:
            ratio = float(min(1.0, p[token] / q_token))
        else:
            ratio = 1.0
        draw = torch.rand((), device=p.device, generator=generator).item()
        if draw < ratio:
            accepted.append(int(token))
            continue
        residual = (p - q).clamp_min(0)
        if residual.sum() <= 0:
            correction = int(torch.multinomial(p, 1, generator=generator).item())
        else:
            correction = int(
                torch.multinomial(residual / residual.sum(), 1, generator=generator).item()
            )
        return SpeculativeResult(accepted, correction, i)
    return SpeculativeResult(accepted, None, None)


def choose_frontier_token(
    result: SpeculativeResult,
    next_target_probs: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> tuple[int, str]:
    """Return the exact token that advances the sequence after a verified block.

    On rejection this is the rejection-sampling correction already drawn by
    ``exact_spec_accept``. When the whole draft block is accepted, canonical
    speculative decoding emits one additional *bonus* token sampled directly from
    the verifier distribution after the last proposal. Both cases intentionally
    enter the same pending-frontier cache path: their verifier KV is materialized
    only on the next target forward.
    """
    if result.correction is not None:
        return int(result.correction), "correction"
    if result.rejected_at is not None:
        raise ValueError("inconsistent speculative result: rejection without correction")
    probs = next_target_probs
    if probs.ndim == 2 and probs.shape[0] == 1:
        probs = probs[0]
    if probs.ndim != 1:
        raise ValueError("next_target_probs must have shape [vocab] or [1, vocab]")
    p = _normalized(probs)
    token = int(torch.multinomial(p, 1, generator=generator).item())
    return token, "bonus"


class ExactSpeculativeDecoder:
    """Thin orchestration shell around a model adapter.

    The adapter must implement ``propose(cache, token_ids, gamma)`` and
    ``verify(cache, token_ids)``. This generic shell owns accept/reject decisions;
    the Qwen HF runtime additionally implements the canonical all-accepted bonus
    token and verifier-anchored pending-frontier cache semantics.
    """

    def __init__(
        self,
        adapter,
        *,
        gamma: int = 4,
        generator: torch.Generator | None = None,
    ):
        if gamma < 1:
            raise ValueError("gamma must be positive")
        self.adapter = adapter
        self.gamma = gamma
        self.generator = generator

    def generate(self, prompt_ids: Sequence[int], max_new_tokens: int) -> list[int]:
        target_cache = self.adapter.target_prefill(prompt_ids)
        anchored = self.adapter.make_anchored_cache(target_cache)
        output: list[int] = []
        while len(output) < max_new_tokens:
            proposal = self.adapter.propose(
                anchored.draft_cache, output[-self.gamma :], self.gamma
            )
            result = exact_spec_accept(
                proposal.token_ids,
                proposal.draft_probs,
                proposal.target_probs,
                generator=self.generator,
            )
            accepted_n = result.accepted_length
            anchored.append_verified(proposal.target_kv.slice(0, accepted_n))
            output.extend(result.accepted)
            if result.correction is not None and len(output) < max_new_tokens:
                output.append(result.correction)
                native = self.adapter.draft_frontier(
                    anchored.draft_cache, result.correction
                )
                anchored.append_pending(result.correction, native)
            if not result.accepted and result.correction is None:
                raise RuntimeError("speculative round made no progress")
        return output[:max_new_tokens]
