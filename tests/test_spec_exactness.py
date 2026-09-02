import torch

from verifier_anchored_sd.spec_decode.exact_sd import exact_spec_accept


def test_matching_distributions_always_accept():
    p = torch.tensor([[0.2, 0.8], [0.7, 0.3]])
    result = exact_spec_accept([1, 0], p, p, generator=torch.Generator().manual_seed(0))
    assert result.accepted == [1, 0]
    assert result.correction is None


def test_correction_is_in_residual_support():
    q = torch.tensor([[0.9, 0.1]])
    p = torch.tensor([[0.1, 0.9]])
    result = exact_spec_accept([0], q, p, generator=torch.Generator().manual_seed(1))
    assert result.accepted == []
    assert result.correction == 1

