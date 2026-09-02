"""State transitions for exact verifier-anchored cache refresh."""

from __future__ import annotations

from dataclasses import dataclass

from .cache_state import CacheState
from .target_to_draft_mapper import RidgeKVMapper


@dataclass
class PendingFrontier:
    token_id: int
    cache_index: int


class VerifierAnchoredCache:
    """Maintains a draft cache whose committed tokens come from the verifier.

    At most one draft-native token is allowed: the correction token, before the
    target has processed it.  The next target verification replaces that slot in
    place, so cache length and positional indices never drift.
    """

    def __init__(self, target_cache: CacheState, mapper: RidgeKVMapper) -> None:
        self.mapper = mapper
        self.draft_cache = mapper.map(target_cache)
        self.pending: PendingFrontier | None = None

    @property
    def seq_len(self) -> int:
        return self.draft_cache.seq_len

    def append_verified(self, target_tokens: CacheState) -> None:
        if self.pending is not None:
            raise RuntimeError("materialize the pending frontier before appending verified KV")
        self.draft_cache.append(self.mapper.map(target_tokens))

    def append_pending(self, token_id: int, native_draft_token: CacheState) -> None:
        if self.pending is not None:
            raise RuntimeError("only one pending frontier is permitted")
        if native_draft_token.seq_len != 1:
            raise ValueError("pending frontier must contain exactly one token")
        self.draft_cache.append(native_draft_token)
        self.pending = PendingFrontier(token_id, self.draft_cache.seq_len - 1)

    def materialize_pending(self, token_id: int, target_token: CacheState) -> None:
        if self.pending is None:
            raise RuntimeError("no pending frontier to materialize")
        if token_id != self.pending.token_id:
            raise ValueError(f"expected pending token {self.pending.token_id}, got {token_id}")
        if target_token.seq_len != 1:
            raise ValueError("target frontier must contain exactly one token")
        self.draft_cache.replace_slice(self.pending.cache_index, self.mapper.map(target_token))
        self.pending = None

