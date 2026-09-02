"""Differentiable acceptance surrogates used to train the small residual adapter."""

from __future__ import annotations

import torch


def total_variation(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Per-position TV distance for distributions shaped ``[..., vocab]``."""
    if p.shape != q.shape:
        raise ValueError("p and q must have identical shapes")
    return 0.5 * (p - q).abs().sum(dim=-1)


def block_acceptance_loss(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    *,
    normalize: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return ``-mean(sum_j prod_{i<=j} alpha_i)`` and diagnostics.

    Inputs are ``[batch, gamma, vocab]``.  ``alpha=1-TV`` is the exact expected
    one-step acceptance mass under rejection sampling.  Products encode the fact
    that later positions only matter when every earlier proposal was accepted.
    """
    if target_probs.ndim != 3 or draft_probs.shape != target_probs.shape:
        raise ValueError("probabilities must both have shape [batch, gamma, vocab]")
    alpha = (1.0 - total_variation(target_probs, draft_probs)).clamp(0.0, 1.0)
    prefix = torch.cumprod(alpha, dim=1)
    expected_length = prefix.sum(dim=1)
    loss = -expected_length.mean()
    if normalize:
        loss = loss / target_probs.shape[1]
    return loss, {"alpha": alpha.detach(), "expected_length": expected_length.detach()}

