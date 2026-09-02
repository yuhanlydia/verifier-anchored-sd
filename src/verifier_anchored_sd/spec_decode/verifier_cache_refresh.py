"""State transitions for exact verifier-anchored cache refresh."""

from __future__ import annotations

from dataclasses import dataclass

from .cache_state import CacheState, RotaryFactors
from .target_to_draft_mapper import RidgeKVMapper


@dataclass
class PendingFrontier:
    token_id: int
    cache_index: int


class VerifierAnchoredCache:
    """Maintain the draft cache with optional verifier-derived refreshes."""

    def __init__(
        self,
        target_cache: CacheState,
        mapper: RidgeKVMapper,
        draft_rotary: RotaryFactors | None = None,
    ) -> None:
        self.mapper = mapper
        self.draft_cache = mapper.map(target_cache, draft_rotary=draft_rotary)
        self.pending: PendingFrontier | None = None

    @classmethod
    def from_native(cls, draft_cache: CacheState, mapper: RidgeKVMapper) -> VerifierAnchoredCache:
        """Construct the Native-SD baseline without paying an unnecessary map."""
        state = cls.__new__(cls)
        state.mapper = mapper
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
        self.draft_cache.append(self.mapper.map(target_tokens, draft_rotary=draft_rotary))

    def append_pending(self, token_id: int, native_draft_token: CacheState) -> None:
        if self.pending is not None:
            raise RuntimeError("only one pending frontier is permitted")
        if native_draft_token.seq_len != 1:
            raise ValueError("pending frontier must contain exactly one token")
        self.draft_cache.append(native_draft_token)
        self.pending = PendingFrontier(token_id, self.draft_cache.seq_len - 1)

    def materialize_pending(
        self,
        token_id: int,
        target_token: CacheState,
        draft_rotary: RotaryFactors | None = None,
    ) -> None:
        if self.pending is None:
            raise RuntimeError("no pending frontier to materialize")
        if token_id != self.pending.token_id:
            raise ValueError(f"expected pending token {self.pending.token_id}, got {token_id}")
        if target_token.seq_len != 1:
            raise ValueError("target frontier must contain exactly one token")
        replacement = self.mapper.map(target_token, draft_rotary=draft_rotary)
        self.draft_cache.replace_slice(self.pending.cache_index, replacement)
        self.pending = None
