import pytest
import torch

from verifier_anchored_sd.subspace_stats import (
    ActivationCovarianceStats,
    BenefitCrossStats,
    GradientCovarianceStats,
    benefit_basis,
    pca_basis,
    sensitivity_basis,
)


def _axis_sign_invariant(vector: torch.Tensor, expected: torch.Tensor) -> bool:
    vector = vector / vector.norm()
    expected = expected / expected.norm()
    return bool(torch.allclose(vector.abs(), expected.abs(), atol=1e-6, rtol=1e-6))


def test_gradient_covariance_recovers_known_sensitive_axis():
    stats = GradientCovarianceStats(prefix_shape=(1, 1, 1), dim=2)
    stats.update(torch.tensor([[[[[2.0, 0.0], [1.0, 0.0], [-3.0, 0.0]]]]]))

    solution = sensitivity_basis(stats, max_rank=2)

    assert solution.vectors.shape == (1, 1, 1, 2, 2)
    assert _axis_sign_invariant(solution.vectors[0, 0, 0, :, 0], torch.tensor([1.0, 0.0]))
    assert solution.eigenvalues[0, 0, 0, 0] > solution.eigenvalues[0, 0, 0, 1]
    assert solution.valid_ranks.item() == 2


def test_benefit_basis_separates_helpful_and_harmful_teacher_directions():
    stats = BenefitCrossStats(prefix_shape=(1, 1, 1), dim=2)
    # Observation 1: g=-e1, Delta=+e1 => beneficial first-order direction (+ eigenvalue).
    # Observation 2: g=+e2, Delta=+e2 => harmful first-order direction (- eigenvalue).
    grad = torch.tensor([[[[[-1.0, 0.0], [0.0, 1.0]]]]])
    delta = torch.tensor([[[[[1.0, 0.0], [0.0, 1.0]]]]])
    stats.update(grad, delta)

    positive = benefit_basis(stats, max_rank=2, sign="positive")
    negative = benefit_basis(stats, max_rank=2, sign="negative")

    assert positive.valid_ranks.item() == 1
    assert negative.valid_ranks.item() == 1
    assert positive.eigenvalues[0, 0, 0, 0] == pytest.approx(0.5)
    assert negative.eigenvalues[0, 0, 0, 0] == pytest.approx(-0.5)
    assert _axis_sign_invariant(positive.vectors[0, 0, 0, :, 0], torch.tensor([1.0, 0.0]))
    assert _axis_sign_invariant(negative.vectors[0, 0, 0, :, 0], torch.tensor([0.0, 1.0]))


def test_benefit_positive_rank_excludes_zero_and_negative_eigenvalues():
    stats = BenefitCrossStats(prefix_shape=(1, 1, 1), dim=3)
    grad = torch.tensor([[[[[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]]])
    delta = torch.tensor([[[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]]])
    stats.update(grad, delta)

    solution = benefit_basis(stats, max_rank=3, sign="positive", eigenvalue_tol=1e-12)

    assert solution.valid_ranks.item() == 1
    assert solution.eigenvalues[0, 0, 0, 0] > 0
    assert torch.all(solution.eigenvalues[0, 0, 0, 1:] == 0)


def test_centered_activation_pca_ignores_constant_offset():
    stats = ActivationCovarianceStats(prefix_shape=(1, 1, 1), dim=2)
    points = torch.tensor([[[[[-2.0, 7.0], [0.0, 7.0], [2.0, 7.0]]]]])
    stats.update(points)

    solution = pca_basis(stats, max_rank=2)

    assert _axis_sign_invariant(solution.vectors[0, 0, 0, :, 0], torch.tensor([1.0, 0.0]))
    assert solution.eigenvalues[0, 0, 0, 0] > 0
    assert abs(float(solution.eigenvalues[0, 0, 0, 1])) < 1e-12


def test_streaming_updates_match_one_batched_update():
    rows = torch.tensor(
        [[[[[1.0, 2.0], [3.0, -1.0], [-2.0, 4.0], [0.5, -0.5]]]]]
    )
    one = GradientCovarianceStats(prefix_shape=(1, 1, 1), dim=2)
    streamed = GradientCovarianceStats(prefix_shape=(1, 1, 1), dim=2)

    one.update(rows)
    streamed.update(rows[..., :2, :])
    streamed.update(rows[..., 2:, :])

    assert torch.allclose(one.matrix(), streamed.matrix(), atol=1e-12, rtol=1e-12)
    assert torch.equal(one.counts, streamed.counts)


def test_activation_streaming_centered_covariance_matches_direct_computation():
    rows = torch.tensor(
        [[[[[1.0, 2.0], [3.0, 0.0], [-1.0, 4.0], [2.0, -2.0]]]]],
        dtype=torch.float64,
    )
    stats = ActivationCovarianceStats(prefix_shape=(1, 1, 1), dim=2)
    stats.update(rows[..., :1, :])
    stats.update(rows[..., 1:, :])

    flat = rows[0, 0, 0]
    centered = flat - flat.mean(dim=0, keepdim=True)
    expected = centered.T @ centered / flat.shape[0]

    assert torch.allclose(stats.matrix()[0, 0, 0], expected, atol=1e-12, rtol=1e-12)


def test_statistics_reject_shape_mismatch_and_nonfinite_values():
    stats = GradientCovarianceStats(prefix_shape=(2, 1, 1), dim=3)
    with pytest.raises(ValueError, match="shape"):
        stats.update(torch.zeros(1, 1, 1, 4, 3))
    with pytest.raises(ValueError, match="finite"):
        bad = torch.zeros(2, 1, 1, 4, 3)
        bad[0, 0, 0, 0, 0] = float("nan")
        stats.update(bad)


def test_basis_requires_observations_and_valid_rank():
    stats = GradientCovarianceStats(prefix_shape=(1, 1, 1), dim=2)
    with pytest.raises(RuntimeError, match="observations"):
        sensitivity_basis(stats, max_rank=1)
    stats.update(torch.ones(1, 1, 1, 1, 2))
    with pytest.raises(ValueError, match="max_rank"):
        sensitivity_basis(stats, max_rank=3)
