"""State transitions for exact verifier-anchored cache refresh."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .cache_state import CacheState, RotaryFactors
from .target_to_draft_mapper import RidgeKVMapper


@dataclass
class PendingFrontier:
    token_id: int
    cache_index: int
    next_probs: torch.Tensor | None = None


class VerifierAnchoredCache:
    """Maintain the draft cache with optional verifier-derived refreshes."""

    def __init__(
        self,
        target_cache: CacheState,
        mapper: RidgeKVMapper,
        draft_rotary: RotaryFactors | None = None,
        output_device: torch.device | str | None = None,
    ) -> None:
        self.mapper = mapper
        self.output_device = output_device
        source = target_cache if output_device is None else target_cache.to(output_device)
        factors = (
            draft_rotary
            if output_device is None or draft_rotary is None
            else draft_rotary.to(output_device)
        )
        self.draft_cache = mapper.map(source, draft_rotary=factors)
        self.pending: PendingFrontier | None = None

    @classmethod
    def from_native(cls, draft_cache: CacheState, mapper: RidgeKVMapper) -> VerifierAnchoredCache:
        """Construct the Native-SD baseline without paying an unnecessary map."""
        state = cls.__new__(cls)
        state.mapper = mapper
        state.output_device = None
        state.draft_cache = draft_cache
        state.pending = None
        return state

    @property
    def seq_len(self) -> int:
        return self.draft_cache.seq_len

    def append_verified(
        self,
        target_tokens: CacheState,
        draft_rotary: RotaryFactors | None = None,
    ) -> None:
        if self.pending is not None:
            raise RuntimeError("materialize the pending frontier before appending verified KV")
        source = target_tokens if self.output_device is None else target_tokens.to(self.output_device)
        factors = (
            draft_rotary
            if self.output_device is None or draft_rotary is None
            else draft_rotary.to(self.output_device)
        )
        self.draft_cache.append(self.mapper.map(source, draft_rotary=factors))

    def append_pending(
        self,
        token_id: int,
        native_draft_token: CacheState,
        *,
        next_probs: torch.Tensor | None = None,
    ) -> None:
        if self.pending is not None:
            raise RuntimeError("only one pending frontier is permitted")
        if native_draft_token.seq_len != 1:
            raise ValueError("pending frontier must contain exactly one token")
        self.draft_cache.append(native_draft_token)
        self.pending = PendingFrontier(
            token_id,
            self.draft_cache.seq_len - 1,
            None if next_probs is None else next_probs.detach().clone(),
        )

    def resolve_pending_native(self, token_id: int) -> torch.Tensor | None:
        """Clear the marker while permanently retaining its native draft KV."""
        if self.pending is None:
            raise RuntimeError("no pending frontier to resolve")
        if token_id != self.pending.token_id:
            raise ValueError(f"expected pending token {self.pending.token_id}, got {token_id}")
        next_probs = self.pending.next_probs
        self.pending = None
        return next_probs

    def materialize_pending(
        self,
        token_id: int,
        target_token: CacheState,
        draft_rotary: RotaryFactors | None = None,
    ) -> torch.Tensor | None:
        if self.pending is None:
            raise RuntimeError("no pending frontier to materialize")
        if token_id != self.pending.token_id:
            raise ValueError(f"expected pending token {self.pending.token_id}, got {token_id}")
        if target_token.seq_len != 1:
            raise ValueError("target frontier must contain exactly one token")
        source = target_token if self.output_device is None else target_token.to(self.output_device)
        factors = (
            draft_rotary
            if self.output_device is None or draft_rotary is None
            else draft_rotary.to(self.output_device)
        )
        replacement = self.mapper.map(source, draft_rotary=factors)
        self.draft_cache.replace_slice(self.pending.cache_index, replacement)
        next_probs = self.pending.next_probs
        self.pending = None
        return next_probs
