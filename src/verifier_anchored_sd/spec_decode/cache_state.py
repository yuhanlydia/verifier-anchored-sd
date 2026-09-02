"""Explicit KV-cache and rotary-position state used by the runtime and tests.

A cache layer follows the Hugging Face convention ``[batch, kv_heads, sequence,
head_dim]``.  Keys are normally stored after RoPE.  Paper-faithful cross-model
translation temporarily moves keys to position-free content space, applies the
linear map there, then applies the receiver model's RoPE factors.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch


def _rotate_half(x: torch.Tensor, *, interleaved: bool) -> torch.Tensor:
    if x.shape[-1] % 2:
        raise ValueError("RoPE requires an even head dimension")
    if interleaved:
        paired = x.reshape(*x.shape[:-1], -1, 2)
        rotated = torch.stack((-paired[..., 1], paired[..., 0]), dim=-1)
        return rotated.flatten(-2)
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


@dataclass(frozen=True)
class RotaryFactors:
    """Model-produced RoPE cosine/sine factors for a contiguous token range."""

    cos: torch.Tensor
    sin: torch.Tensor
    interleaved: bool = False

    def __post_init__(self) -> None:
        if self.cos.shape != self.sin.shape:
            raise ValueError("RoPE cosine and sine tensors must have identical shapes")
        if self.cos.ndim not in {2, 3}:
            raise ValueError("RoPE factors must be [tokens, dim] or [batch, tokens, dim]")

    @property
    def seq_len(self) -> int:
        return self.cos.shape[-2]

    def apply(self, x: torch.Tensor, *, inverse: bool = False) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("RoPE input must be [batch, heads, tokens, dim]")
        cos = self.cos.to(device=x.device, dtype=x.dtype)
        sin = self.sin.to(device=x.device, dtype=x.dtype)
        if cos.ndim == 2:
            cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
        if cos.shape[-2:] != x.shape[-2:]:
            raise ValueError(
                f"RoPE factors {tuple(cos.shape)} do not match cache {tuple(x.shape)}"
            )
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        sign = -1.0 if inverse else 1.0
        return x * cos + sign * _rotate_half(x, interleaved=self.interleaved) * sin

    def slice(self, start: int, end: int | None = None) -> RotaryFactors:
        end = self.seq_len if end is None else end
        if not 0 <= start <= end <= self.seq_len:
            raise ValueError("invalid RoPE slice")
        return RotaryFactors(
            self.cos[..., start:end, :], self.sin[..., start:end, :], self.interleaved
        )

    def append(self, other: RotaryFactors) -> RotaryFactors:
        if self.interleaved != other.interleaved or self.cos.ndim != other.cos.ndim:
            raise ValueError("incompatible RoPE factors")
        if self.cos.shape[:-2] != other.cos.shape[:-2] or self.cos.shape[-1] != other.cos.shape[-1]:
            raise ValueError("incompatible RoPE factor shapes")
        return RotaryFactors(
            torch.cat((self.cos, other.cos), dim=-2),
            torch.cat((self.sin, other.sin), dim=-2),
            self.interleaved,
        )

    def clone(self) -> RotaryFactors:
        return RotaryFactors(self.cos.clone(), self.sin.clone(), self.interleaved)

    def to(self, *args, **kwargs) -> RotaryFactors:
        return RotaryFactors(self.cos.to(*args, **kwargs), self.sin.to(*args, **kwargs), self.interleaved)


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
    """A mutable, layer-major KV cache with explicit RoPE provenance."""

    def __init__(
        self,
        layers: Iterable[LayerKV] = (),
        *,
        rotary: RotaryFactors | None = None,
        keys_are_content: bool = False,
    ) -> None:
        self.layers = list(layers)
        self.rotary = rotary
        self.keys_are_content = keys_are_content
        for layer in self.layers:
            layer.validate()
        if self.layers and len({x.seq_len for x in self.layers}) != 1:
            raise ValueError("all cache layers must have the same sequence length")
        if rotary is not None and self.layers and rotary.seq_len != self.seq_len:
            raise ValueError("RoPE factors do not match cache sequence length")
        if keys_are_content and rotary is None:
            # A content-space cache may be useful without factors after fitting, but runtime
            # caches must carry factors.  Keep this permissive for mapper intermediates.
            pass

    @classmethod
    def from_tuple(
        cls,
        past_key_values: Sequence[tuple[torch.Tensor, torch.Tensor]],
        *,
        rotary: RotaryFactors | None = None,
    ) -> CacheState:
        return cls((LayerKV(k, v) for k, v in past_key_values), rotary=rotary)

    def as_tuple(self) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        return tuple((x.key, x.value) for x in self.layers)

    def clone(self) -> CacheState:
        return CacheState(
            (LayerKV(x.key.clone(), x.value.clone()) for x in self.layers),
            rotary=None if self.rotary is None else self.rotary.clone(),
            keys_are_content=self.keys_are_content,
        )

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    @property
    def seq_len(self) -> int:
        return self.layers[0].seq_len if self.layers else 0

    @property
    def kv_heads(self) -> int:
        return self.layers[0].key.shape[1] if self.layers else 0

    @property
    def head_dim(self) -> int:
        return self.layers[0].key.shape[-1] if self.layers else 0

    def append(self, other: CacheState) -> None:
        if self.num_layers != other.num_layers:
            raise ValueError("cannot append caches with different layer counts")
        if self.keys_are_content != other.keys_are_content:
            raise ValueError("cannot append position-space and content-space caches")
        if (self.rotary is None) != (other.rotary is None):
            raise ValueError("cannot append caches with mismatched RoPE provenance")
        for dst, src in zip(self.layers, other.layers):
            if dst.key.shape[:2] != src.key.shape[:2] or dst.key.shape[-1] != src.key.shape[-1]:
                raise ValueError("cannot append caches with incompatible batch/head dimensions")
            dst.key = torch.cat((dst.key, src.key), dim=-2)
            dst.value = torch.cat((dst.value, src.value), dim=-2)
        if self.rotary is not None and other.rotary is not None:
            self.rotary = self.rotary.append(other.rotary)

    def truncate(self, length: int) -> CacheState:
        if not 0 <= length <= self.seq_len:
            raise ValueError(f"length {length} outside [0, {self.seq_len}]")
        for layer in self.layers:
            layer.key = layer.key[..., :length, :]
            layer.value = layer.value[..., :length, :]
        if self.rotary is not None:
            self.rotary = self.rotary.slice(0, length)
        return self

    def slice(self, start: int, end: int | None = None) -> CacheState:
        end = self.seq_len if end is None else end
        if not 0 <= start <= end <= self.seq_len:
            raise ValueError("invalid cache slice")
        return CacheState(
            (LayerKV(x.key[..., start:end, :], x.value[..., start:end, :]) for x in self.layers),
            rotary=None if self.rotary is None else self.rotary.slice(start, end),
            keys_are_content=self.keys_are_content,
        )

    def to(self, *args, **kwargs) -> CacheState:
        return CacheState(
            (LayerKV(x.key.to(*args, **kwargs), x.value.to(*args, **kwargs)) for x in self.layers),
            rotary=None if self.rotary is None else self.rotary.to(*args, **kwargs),
            keys_are_content=self.keys_are_content,
        )

    def to_content_space(self) -> CacheState:
        if self.keys_are_content:
            return self
        if self.rotary is None:
            raise ValueError("content-space KV mapping requires source RoPE factors")
        return CacheState(
            (LayerKV(self.rotary.apply(x.key, inverse=True), x.value) for x in self.layers),
            rotary=self.rotary,
            keys_are_content=True,
        )

    def apply_rotary(self, factors: RotaryFactors) -> CacheState:
        if not self.keys_are_content:
            raise ValueError("cannot apply target RoPE to keys already in position space")
        if factors.seq_len != self.seq_len:
            raise ValueError("target RoPE factors do not match cache length")
        return CacheState(
            (LayerKV(factors.apply(x.key), x.value) for x in self.layers),
            rotary=factors,
            keys_are_content=False,
        )

    def replace_slice(self, start: int, replacement: CacheState) -> None:
        """Replace a contiguous range without changing cache length or positions."""
        end = start + replacement.seq_len
        if start < 0 or end > self.seq_len or self.num_layers != replacement.num_layers:
            raise ValueError("replacement does not fit in cache")
        if self.keys_are_content != replacement.keys_are_content:
            raise ValueError("replacement cache uses a different key space")
        for dst, src in zip(self.layers, replacement.layers):
            if dst.key.shape[:2] != src.key.shape[:2] or dst.key.shape[-1] != src.key.shape[-1]:
                raise ValueError("replacement has incompatible dimensions")
            dst.key[..., start:end, :] = src.key
            dst.value[..., start:end, :] = src.value
