"""Simple differentiable scribble coverage loss (not the full ScribbleDiff objective)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def scribble_target(mask: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Convert [1,1,H,W] stroke mask to a distribution on the image-token grid."""
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError("Expected scribble mask [B,1,H,W]")
    # Max pooling avoids losing thin drawn strokes when reducing to 32x32 tokens.
    target = F.adaptive_max_pool2d(mask.float(), (h, w)).flatten(1)
    if not (target.sum(dim=-1) > 0).all():
        raise ValueError("The scribble contains no visible pixels after processing")
    return target / target.sum(dim=-1, keepdim=True)


def attention_coverage_loss(attn: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cross entropy between stroke distribution and spatial object-token attention.

    attn: [B, H*W] image->object-text attention (aggregated over heads/blocks).
    target: [B, H*W] spatial distribution from the scribble.
    """
    if attn.shape != target.shape:
        raise ValueError(f"Attention shape {attn.shape} != scribble shape {target.shape}")
    attn = attn.float().clamp_min(1e-12)
    attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return -(target.float() * attn.log()).sum(dim=-1).mean()


def normalized_latent_update(
    latent: torch.Tensor, grad: torch.Tensor, strength: float
) -> tuple[torch.Tensor, float]:
    """RMS-scaled gradient descent on latent; returns detached latent and gradient RMS."""
    rms = grad.float().square().mean().sqrt().clamp_min(1e-8)
    updated = (latent.float() - strength * grad.float() / rms).detach()
    return updated, rms.item()
