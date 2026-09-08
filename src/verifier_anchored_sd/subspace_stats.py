"""Streaming sufficient statistics for student-readable KV subspaces.

All statistics are accumulated on CPU in float64.  The intended prefix geometry is
``[draft_layers, 2, kv_heads]`` (K/V kind axis is 2), while the final two axes of
an update tensor are ``[observations, head_dim]``.  No per-token gradient rows are
retained after an update.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(frozen=True)
class BasisSolution:
    """Ordered basis vectors/eigenvalues plus usable rank per prefix group."""

    vectors: torch.Tensor
    eigenvalues: torch.Tensor
    valid_ranks: torch.Tensor


class _BaseStats:
    def __init__(self, *, prefix_shape: tuple[int, ...], dim: int) -> None:
        if not prefix_shape or any(int(size) <= 0 for size in prefix_shape):
            raise ValueError("prefix_shape must contain positive dimensions")
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.prefix_shape = tuple(int(size) for size in prefix_shape)
        self.dim = int(dim)
        self.counts = torch.zeros(self.prefix_shape, dtype=torch.int64, device="cpu")

    @property
    def expected_rank(self) -> int:
        return len(self.prefix_shape) + 2

    def _rows(self, values: torch.Tensor, *, name: str) -> torch.Tensor:
        if not isinstance(values, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        expected_prefix = self.prefix_shape
        if values.ndim != self.expected_rank:
            raise ValueError(
                f"{name} shape must be prefix_shape + [observations, dim]; "
                f"got {tuple(values.shape)}"
            )
        if tuple(values.shape[: len(expected_prefix)]) != expected_prefix or values.shape[-1] != self.dim:
            raise ValueError(
                f"{name} shape must be {expected_prefix} + [observations, {self.dim}]; "
                f"got {tuple(values.shape)}"
            )
        if values.shape[-2] <= 0:
            raise ValueError(f"{name} must contain at least one observation")
        rows = values.detach().to(device="cpu", dtype=torch.float64)
        if not torch.isfinite(rows).all():
            raise ValueError(f"{name} values must be finite")
        return rows

    def _increment_counts(self, observations: int) -> None:
        self.counts += int(observations)

    def _require_observations(self) -> None:
        if (self.counts <= 0).any():
            raise RuntimeError("subspace statistics contain groups with no observations")


class GradientCovarianceStats(_BaseStats):
    """Accumulate ``sum g g^T`` for decoder-sensitivity subspaces."""

    def __init__(self, *, prefix_shape: tuple[int, ...], dim: int) -> None:
        super().__init__(prefix_shape=prefix_shape, dim=dim)
        self.second = torch.zeros(
            (*self.prefix_shape, self.dim, self.dim), dtype=torch.float64, device="cpu"
        )

    def update(self, gradients: torch.Tensor) -> None:
        rows = self._rows(gradients, name="gradient")
        self.second += torch.einsum("...nd,...ne->...de", rows, rows)
        self._increment_counts(rows.shape[-2])

    def matrix(self) -> torch.Tensor:
        self._require_observations()
        return self.second / self.counts.to(torch.float64)[..., None, None]


class BenefitCrossStats(_BaseStats):
    r"""Accumulate the symmetric first-order teacher-delta benefit matrix.

    The matrix is

    ``S = -0.5 E[g Delta^T + Delta g^T]``.

    For any orthogonal rank-r projector ``P``, ``trace(P S)`` equals the negative
    first-order change ``-E[g^T P Delta]`` predicted from injecting the projected
    teacher delta. Positive eigenvalues are therefore predicted-helpful directions;
    negative eigenvalues are signed harmful controls.
    """

    def __init__(self, *, prefix_shape: tuple[int, ...], dim: int) -> None:
        super().__init__(prefix_shape=prefix_shape, dim=dim)
        self.cross = torch.zeros(
            (*self.prefix_shape, self.dim, self.dim), dtype=torch.float64, device="cpu"
        )

    def update(self, gradients: torch.Tensor, deltas: torch.Tensor) -> None:
        g = self._rows(gradients, name="gradient")
        delta = self._rows(deltas, name="delta")
        if g.shape != delta.shape:
            raise ValueError("gradient and delta shapes must match")
        gd = torch.einsum("...nd,...ne->...de", g, delta)
        dg = torch.einsum("...nd,...ne->...de", delta, g)
        self.cross += -0.5 * (gd + dg)
        self._increment_counts(g.shape[-2])

    def matrix(self) -> torch.Tensor:
        self._require_observations()
        matrix = self.cross / self.counts.to(torch.float64)[..., None, None]
        # Numerical safety: eigensolvers below assume exact symmetry.
        return 0.5 * (matrix + matrix.transpose(-1, -2))


class ActivationCovarianceStats(_BaseStats):
    """Accumulate centered mapped-KV covariance without retaining activation rows."""

    def __init__(self, *, prefix_shape: tuple[int, ...], dim: int) -> None:
        super().__init__(prefix_shape=prefix_shape, dim=dim)
        self.sum = torch.zeros((*self.prefix_shape, self.dim), dtype=torch.float64, device="cpu")
        self.second = torch.zeros(
            (*self.prefix_shape, self.dim, self.dim), dtype=torch.float64, device="cpu"
        )

    def update(self, activations: torch.Tensor) -> None:
        rows = self._rows(activations, name="activation")
        self.sum += rows.sum(dim=-2)
        self.second += torch.einsum("...nd,...ne->...de", rows, rows)
        self._increment_counts(rows.shape[-2])

    def matrix(self) -> torch.Tensor:
        self._require_observations()
        counts = self.counts.to(torch.float64)
        mean = self.sum / counts[..., None]
        second_moment = self.second / counts[..., None, None]
        covariance = second_moment - torch.einsum("...d,...e->...de", mean, mean)
        covariance = 0.5 * (covariance + covariance.transpose(-1, -2))
        # Centered covariance is PSD in exact arithmetic. Clamp only tiny diagonal/
        # eigensolver noise later rather than changing the matrix orientation here.
        return covariance


def _validate_rank(stats: _BaseStats, max_rank: int) -> int:
    stats._require_observations()
    if not 0 < int(max_rank) <= stats.dim:
        raise ValueError("max_rank must lie in [1, dim]")
    return int(max_rank)


def _descending_eigenbasis(matrix: torch.Tensor, max_rank: int) -> BasisSolution:
    values, vectors = torch.linalg.eigh(matrix)
    order = torch.arange(values.shape[-1] - 1, -1, -1, device=values.device)
    values = values.index_select(-1, order)[..., :max_rank]
    vectors = vectors.index_select(-1, order)[..., :, :max_rank]
    # PSD matrices can receive tiny negative values from roundoff. They remain
    # valid sensitivity/PCA directions; expose zero rather than a fake negative.
    values = values.clamp_min(0.0)
    valid = torch.full(values.shape[:-1], max_rank, dtype=torch.int64, device="cpu")
    return BasisSolution(vectors.float().cpu(), values.cpu(), valid)


def sensitivity_basis(stats: GradientCovarianceStats, *, max_rank: int) -> BasisSolution:
    """Return top eigendirections of ``E[g g^T]`` in descending sensitivity."""
    rank = _validate_rank(stats, max_rank)
    return _descending_eigenbasis(stats.matrix(), rank)


def pca_basis(stats: ActivationCovarianceStats, *, max_rank: int) -> BasisSolution:
    """Return top eigendirections of centered mapped-activation covariance."""
    rank = _validate_rank(stats, max_rank)
    return _descending_eigenbasis(stats.matrix(), rank)


def benefit_basis(
    stats: BenefitCrossStats,
    *,
    max_rank: int,
    sign: Literal["positive", "negative"] = "positive",
    eigenvalue_tol: float = 1e-10,
) -> BasisSolution:
    """Return signed eigendirections of the first-order benefit matrix.

    Positive mode keeps only eigenvalues ``> eigenvalue_tol`` ordered from most
    positive down. Negative mode keeps only values ``< -eigenvalue_tol`` ordered
    from most negative upward (largest predicted harm first). Unused output columns
    and eigenvalues are zero-padded so every family has the same max-rank shape.
    """
    rank = _validate_rank(stats, max_rank)
    if sign not in {"positive", "negative"}:
        raise ValueError("sign must be 'positive' or 'negative'")
    if not torch.isfinite(torch.tensor(float(eigenvalue_tol))) or eigenvalue_tol < 0:
        raise ValueError("eigenvalue_tol must be finite and non-negative")

    matrix = stats.matrix()
    raw_values, raw_vectors = torch.linalg.eigh(matrix)  # ascending
    prefix = raw_values.shape[:-1]
    output_vectors = torch.zeros((*prefix, stats.dim, rank), dtype=torch.float64)
    output_values = torch.zeros((*prefix, rank), dtype=torch.float64)
    valid = torch.zeros(prefix, dtype=torch.int64)

    # Prefix group count is small (36*2*8 for Qwen3-4B). A transparent loop keeps
    # sign filtering and effective-rank semantics unambiguous.
    for index in torch.cartesian_prod(*[torch.arange(size) for size in prefix]):
        key = tuple(int(value) for value in index.tolist()) if index.ndim else (int(index),)
        values = raw_values[key]
        vectors = raw_vectors[key]
        if sign == "positive":
            selected = torch.nonzero(values > eigenvalue_tol, as_tuple=False).flatten()
            selected = selected.flip(0)  # largest positive first
        else:
            selected = torch.nonzero(values < -eigenvalue_tol, as_tuple=False).flatten()
            # eigh is ascending, so most negative is already first.
        selected = selected[:rank]
        count = int(selected.numel())
        valid[key] = count
        if count:
            output_values[key + (slice(0, count),)] = values[selected]
            output_vectors[key + (slice(None), slice(0, count))] = vectors[:, selected]

    return BasisSolution(output_vectors.float(), output_values, valid)
