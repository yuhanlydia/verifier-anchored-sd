"""Differentiable acceptance surrogates for the low-rank mapper residual."""

from __future__ import annotations

import torch


def total_variation(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Per-position TV distance for distributions shaped ``[..., vocab]``."""
    if p.shape != q.shape:
        raise ValueError("p and q must have identical shapes")
    return 0.5 * (p - q).abs().sum(dim=-1)


def acceptance_mass(
    target_probs: torch.Tensor, draft_probs: torch.Tensor
) -> torch.Tensor:
    """Exact one-position expected speculative acceptance mass ``1-TV(p,q)``."""
    if target_probs.ndim != 3 or draft_probs.shape != target_probs.shape:
        raise ValueError("probabilities must both have shape [batch, gamma, vocab]")
    return (1.0 - total_variation(target_probs, draft_probs)).clamp(0.0, 1.0)


def one_step_acceptance_loss(
    target_probs: torch.Tensor, draft_probs: torch.Tensor
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Baseline that optimizes only the first proposal position."""
    alpha = acceptance_mass(target_probs, draft_probs)
    loss = -alpha[:, 0].mean()
    return loss, {"alpha": alpha.detach(), "first_acceptance": alpha[:, 0].detach()}


def block_acceptance_loss(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    *,
    normalize: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return the prefix-weighted multi-position acceptance surrogate.

    Inputs are ``[batch, gamma, vocab]``. ``alpha=1-TV`` is the exact expected
    one-step acceptance mass at each sampled conditional state. Products encode the
    speculative requirement that a later proposal is reached only when all earlier
    proposals in that block were accepted.
    """
    alpha = acceptance_mass(target_probs, draft_probs)
    prefix = torch.cumprod(alpha, dim=1)
    expected_length = prefix.sum(dim=1)
    loss = -expected_length.mean()
    if normalize:
        loss = loss / target_probs.shape[1]
    return loss, {"alpha": alpha.detach(), "expected_length": expected_length.detach()}
