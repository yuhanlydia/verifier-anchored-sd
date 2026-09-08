import pytest
import torch

from verifier_anchored_sd.kv_subspace import (
    InterventionSpec,
    SubspaceBasisArtifact,
    apply_intervention,
    deterministic_random_basis,
)
from verifier_anchored_sd.spec_decode.cache_state import CacheState, LayerKV, RotaryFactors


def _rotary(tokens: int, dim: int) -> RotaryFactors:
    positions = torch.arange(tokens, dtype=torch.float32).unsqueeze(1)
    freqs = torch.arange(dim, dtype=torch.float32).unsqueeze(0) + 1.0
    angles = 0.07 * positions * freqs
    return RotaryFactors(torch.cos(angles), torch.sin(angles), interleaved=False)


def _cache(values: torch.Tensor, *, rotary: RotaryFactors) -> CacheState:
    # one layer, one head; input values are content-space keys and value tensors
    key_content = values.clone().reshape(1, 1, values.shape[0], values.shape[1])
    value = (values * 0.5).reshape(1, 1, values.shape[0], values.shape[1])
    key_position = rotary.apply(key_content)
    return CacheState([LayerKV(key_position, value)], rotary=rotary, keys_are_content=False)


def _artifact(basis: torch.Tensor, eigenvalues: torch.Tensor, valid_rank: int = 2):
    # basis input [D,R] -> artifact [L,2,H,D,R]
    stacked = basis.view(1, 1, 1, *basis.shape).repeat(1, 2, 1, 1, 1)
    eigs = eigenvalues.view(1, 1, 1, -1).repeat(1, 2, 1, 1)
    ranks = torch.full((1, 2, 1), valid_rank, dtype=torch.long)
    return SubspaceBasisArtifact(
        metadata={
            "schema_version": 1,
            "draft_layers": 1,
            "kv_heads": 1,
            "head_dim": basis.shape[0],
            "max_rank": basis.shape[1],
        },
        bases={"grad": stacked},
        eigenvalues={"grad": eigs},
        valid_ranks={"grad": ranks},
    )


def test_artifact_rejects_non_orthonormal_basis():
    bad = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    artifact = _artifact(bad, torch.tensor([2.0, 1.0]))

    with pytest.raises(ValueError, match="orthonormal"):
        artifact.validate()


def test_soft_beta_one_is_exact_full_mapped_identity():
    rotary = _rotary(3, 2)
    mapped = _cache(torch.tensor([[1.0, 2.0], [2.0, -1.0], [0.5, 1.5]]), rotary=rotary)
    native = _cache(torch.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]), rotary=rotary)
    artifact = _artifact(torch.eye(2), torch.tensor([2.0, 1.0]))

    out = apply_intervention(
        native,
        mapped,
        artifact,
        InterventionSpec(mode="mapped_soft", family="grad", rank=1, beta=1.0),
    )

    for got, expected in zip(out.layers, mapped.layers, strict=True):
        assert torch.allclose(got.key, expected.key, atol=1e-6, rtol=1e-6)
        assert torch.allclose(got.value, expected.value, atol=1e-6, rtol=1e-6)


def test_hard_projection_operates_in_content_space_then_restores_rotary():
    rotary = _rotary(2, 2)
    mapped = _cache(torch.tensor([[2.0, 3.0], [-1.0, 4.0]]), rotary=rotary)
    native = _cache(torch.zeros(2, 2), rotary=rotary)
    artifact = _artifact(torch.eye(2), torch.tensor([2.0, 1.0]))

    out = apply_intervention(
        native,
        mapped,
        artifact,
        InterventionSpec(mode="mapped_soft", family="grad", rank=1, beta=0.0),
    )

    content = out.to_content_space()
    expected_key = torch.tensor([[[[2.0, 0.0], [-1.0, 0.0]]]])
    expected_value = torch.tensor([[[[1.0, 0.0], [-0.5, 0.0]]]])
    assert torch.allclose(content.layers[0].key, expected_key, atol=1e-5, rtol=1e-5)
    assert torch.allclose(content.layers[0].value, expected_value, atol=1e-6, rtol=1e-6)
    assert out.keys_are_content is False
    assert out.rotary is not None


def test_projected_delta_preserves_native_base_and_scales_readable_teacher_delta():
    rotary = _rotary(2, 2)
    native = _cache(torch.tensor([[1.0, 1.0], [1.0, 1.0]]), rotary=rotary)
    mapped = _cache(torch.tensor([[3.0, 5.0], [5.0, 3.0]]), rotary=rotary)
    artifact = _artifact(torch.eye(2), torch.tensor([2.0, 1.0]))

    out = apply_intervention(
        native,
        mapped,
        artifact,
        InterventionSpec(mode="delta_projected", family="grad", rank=1, alpha=0.5),
    ).to_content_space()

    # native + .5 * P(first-dimension mapped-native delta)
    assert torch.allclose(
        out.layers[0].key,
        torch.tensor([[[[2.0, 1.0], [3.0, 1.0]]]]),
        atol=1e-5,
        rtol=1e-5,
    )


def test_delta_shrink_beta_one_recovers_full_mapped():
    rotary = _rotary(2, 2)
    native = _cache(torch.tensor([[1.0, -2.0], [0.0, 1.0]]), rotary=rotary)
    mapped = _cache(torch.tensor([[4.0, 3.0], [-2.0, 2.0]]), rotary=rotary)
    artifact = _artifact(torch.eye(2), torch.tensor([2.0, 1.0]))

    out = apply_intervention(
        native,
        mapped,
        artifact,
        InterventionSpec(mode="delta_shrink", family="grad", rank=1, beta=1.0),
    )

    for got, expected in zip(out.layers, mapped.layers, strict=True):
        assert torch.allclose(got.key, expected.key, atol=1e-6, rtol=1e-6)
        assert torch.allclose(got.value, expected.value, atol=1e-6, rtol=1e-6)


def test_effective_rank_clips_requested_rank_to_valid_positive_directions():
    artifact = _artifact(torch.eye(2), torch.tensor([3.0, -2.0]), valid_rank=1)

    assert artifact.effective_rank("grad", requested_rank=2, layer=0, kind=0, head=0) == 1


def test_deterministic_random_basis_is_orthonormal_and_repeatable():
    a = deterministic_random_basis(
        layers=2, kinds=2, heads=3, head_dim=8, max_rank=4, seed=17
    )
    b = deterministic_random_basis(
        layers=2, kinds=2, heads=3, head_dim=8, max_rank=4, seed=17
    )

    assert torch.equal(a, b)
    for layer in range(2):
        for kind in range(2):
            for head in range(3):
                u = a[layer, kind, head]
                assert torch.allclose(u.T @ u, torch.eye(4), atol=1e-5, rtol=1e-5)


def test_orthogonal_mode_removes_gradient_subspace_instead_of_retaining_it():
    rotary = _rotary(1, 2)
    mapped = _cache(torch.tensor([[2.0, 3.0]]), rotary=rotary)
    native = _cache(torch.zeros(1, 2), rotary=rotary)
    artifact = _artifact(torch.eye(2), torch.tensor([2.0, 1.0]))

    out = apply_intervention(
        native,
        mapped,
        artifact,
        InterventionSpec(mode="orthogonal", family="grad", rank=1),
    ).to_content_space()

    assert torch.allclose(out.layers[0].key, torch.tensor([[[[0.0, 3.0]]]]), atol=1e-5)
