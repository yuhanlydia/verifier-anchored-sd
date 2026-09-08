import pytest
import torch

from verifier_anchored_sd.kv_subspace import InterventionSpec, SubspaceBasisArtifact
from verifier_anchored_sd.spec_decode.cache_state import CacheState, LayerKV, RotaryFactors
from verifier_anchored_sd.spec_decode.subspace_mapper import SubspaceMappedKVMapper


class FakeMapper:
    def __init__(self, mapped: CacheState):
        self._mapped = mapped
        self.metadata = type(
            "Meta",
            (),
            {"draft_layers": 1, "draft_kv_heads": 1, "head_dim": 2},
        )()
        self.device = torch.device("cpu")

    def map(self, target, *, draft_rotary=None, include_residual=True):
        return self._mapped.clone()


def _rotary(tokens=2):
    return RotaryFactors(torch.ones(tokens, 2), torch.zeros(tokens, 2), False)


def _mapped():
    rotary = _rotary()
    return CacheState(
        [
            LayerKV(
                torch.tensor([[[[2.0, 3.0], [4.0, 5.0]]]]),
                torch.tensor([[[[1.0, -2.0], [3.0, 4.0]]]]),
            )
        ],
        rotary=rotary,
        keys_are_content=False,
    )


def _artifact():
    basis = torch.eye(2).view(1, 1, 1, 2, 2).repeat(1, 2, 1, 1, 1)
    eig = torch.ones(1, 2, 1, 2)
    ranks = torch.full((1, 2, 1), 2, dtype=torch.long)
    return SubspaceBasisArtifact(
        metadata={
            "schema_version": 1,
            "draft_layers": 1,
            "kv_heads": 1,
            "head_dim": 2,
            "max_rank": 2,
            "mapper_checkpoint_sha256": "mapper-sha",
        },
        bases={"grad": basis},
        eigenvalues={"grad": eig},
        valid_ranks={"grad": ranks},
    )


def test_beta_one_wrapper_is_exact_base_mapper_identity():
    base = FakeMapper(_mapped())
    wrapper = SubspaceMappedKVMapper(
        base,
        _artifact(),
        InterventionSpec(mode="mapped_soft", family="grad", rank=1, beta=1.0),
    )

    result = wrapper.map(CacheState([]), draft_rotary=_rotary())
    expected = base.map(CacheState([]), draft_rotary=_rotary())

    assert torch.equal(result.layers[0].key, expected.layers[0].key)
    assert torch.equal(result.layers[0].value, expected.layers[0].value)


def test_hard_wrapper_projects_only_mapped_cache_without_native_history():
    wrapper = SubspaceMappedKVMapper(
        FakeMapper(_mapped()),
        _artifact(),
        InterventionSpec(mode="mapped_soft", family="grad", rank=1, beta=0.0),
    )

    result = wrapper.map(CacheState([]), draft_rotary=_rotary()).to_content_space()

    assert torch.allclose(
        result.layers[0].key,
        torch.tensor([[[[2.0, 0.0], [4.0, 0.0]]]]),
    )
    assert torch.allclose(
        result.layers[0].value,
        torch.tensor([[[[1.0, 0.0], [3.0, 0.0]]]]),
    )


def test_wrapper_rejects_delta_intervention_that_requires_native_history():
    with pytest.raises(ValueError, match="deployment"):
        SubspaceMappedKVMapper(
            FakeMapper(_mapped()),
            _artifact(),
            InterventionSpec(mode="delta_projected", family="grad", rank=1, alpha=1.0),
        )


def test_orthogonal_wrapper_keeps_complement_of_readable_basis():
    wrapper = SubspaceMappedKVMapper(
        FakeMapper(_mapped()),
        _artifact(),
        InterventionSpec(mode="orthogonal", family="grad", rank=1),
    )

    result = wrapper.map(CacheState([]), draft_rotary=_rotary()).to_content_space()

    assert torch.allclose(
        result.layers[0].key,
        torch.tensor([[[[0.0, 3.0], [0.0, 5.0]]]]),
    )


def test_wrapper_exposes_base_mapper_metadata_and_device():
    base = FakeMapper(_mapped())
    wrapper = SubspaceMappedKVMapper(
        base,
        _artifact(),
        InterventionSpec(mode="mapped_soft", family="grad", rank=1, beta=0.5),
    )

    assert wrapper.metadata is base.metadata
    assert wrapper.device == base.device
