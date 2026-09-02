"""Small, explicit cache container used by the runtime and by tests.

The project deliberately keeps cache mutation out of model-specific code.  A cache is
represented as one ``(key, value)`` pair per layer, with tensors shaped
``[batch, kv_heads, sequence, head_dim]`` (the Hugging Face convention).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch


@dataclass
class LayerKV:
    key: torch.Tensor
    value: torch.Tensor

    @property
    def seq_len(self) -> int:
        return self.key.shape[-2]

    def validate(self) -> None:
        if self.key.ndim != 4 or self.value.ndim != 4:
            raise ValueError("KV tensors must have shape [batch, heads, sequence, head_dim]")
        if self.key.shape != self.value.shape:
            raise ValueError(f"K/V shape mismatch: {self.key.shape} vs {self.value.shape}")


class CacheState:
    """A mutable, layer-major KV cache with safe append/truncate operations."""

    def __init__(self, layers: Iterable[LayerKV] = ()) -> None:
        self.layers = list(layers)
        for layer in self.layers:
            layer.validate()
        if self.layers and len({x.seq_len for x in self.layers}) != 1:
            raise ValueError("all cache layers must have the same sequence length")

    @classmethod
    def from_tuple(cls, past_key_values: Sequence[tuple[torch.Tensor, torch.Tensor]]) -> CacheState:
        return cls(LayerKV(k, v) for k, v in past_key_values)

    def as_tuple(self) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        return tuple((x.key, x.value) for x in self.layers)

    def clone(self) -> CacheState:
        return CacheState(LayerKV(x.key.clone(), x.value.clone()) for x in self.layers)

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    @property
    def seq_len(self) -> int:
        return self.layers[0].seq_len if self.layers else 0

    def append(self, other: CacheState) -> None:
        if self.num_layers != other.num_layers:
            raise ValueError("cannot append caches with different layer counts")
        for dst, src in zip(self.layers, other.layers):
            if dst.key.shape[:2] != src.key.shape[:2] or dst.key.shape[-1] != src.key.shape[-1]:
                raise ValueError("cannot append caches with incompatible batch/head dimensions")
            dst.key = torch.cat((dst.key, src.key), dim=-2)
            dst.value = torch.cat((dst.value, src.value), dim=-2)

    def truncate(self, length: int) -> CacheState:
        if not 0 <= length <= self.seq_len:
            raise ValueError(f"length {length} outside [0, {self.seq_len}]")
        for layer in self.layers:
            layer.key = layer.key[..., :length, :]
            layer.value = layer.value[..., :length, :]
        return self

    def slice(self, start: int, end: int | None = None) -> CacheState:
        end = self.seq_len if end is None else end
        if not 0 <= start <= end <= self.seq_len:
            raise ValueError("invalid cache slice")
        return CacheState(
            LayerKV(x.key[..., start:end, :], x.value[..., start:end, :]) for x in self.layers
        )

    def to(self, *args, **kwargs) -> CacheState:
        return CacheState(LayerKV(x.key.to(*args, **kwargs), x.value.to(*args, **kwargs)) for x in self.layers)

    def replace_slice(self, start: int, replacement: CacheState) -> None:
        """Replace a contiguous range without changing cache length.

        This is the operation used when a pending correction frontier is replaced by
        the target's exact KV on the next verification round.
        """
        end = start + replacement.seq_len
        if start < 0 or end > self.seq_len or self.num_layers != replacement.num_layers:
            raise ValueError("replacement does not fit in cache")
        for dst, src in zip(self.layers, replacement.layers):
            if dst.key.shape[:2] != src.key.shape[:2] or dst.key.shape[-1] != src.key.shape[-1]:
                raise ValueError("replacement has incompatible dimensions")
            dst.key[..., start:end, :] = src.key
            dst.value[..., start:end, :] = src.value

