"""Online, same-timestep adaptive weighting of FLUX.2 Klein attention heads.

At step t, with each head's image-query -> object-text probability map A_h:
  1. Guide the latent using the current weighted map A = sum_h(w_h A_h).
  2. Re-evaluate the individual head maps at the *updated* latent, at SAME t.
  3. improvement[h] = score(A_h before) - score(A_h after)
  4. w_next = EMA(w_previous, (1-uniform_mix)*softmax(improvement/tau)
                                + uniform_mix/H)

The second map is collected from the conditional forward already required for
Klein's actual denoising velocity; no extra model forward/backward is added.

This measures a head's responsiveness to a common latent perturbation, not
causal attribution or proof that a head controls object shape.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from guidance import _distance_map_to_scribble, _moment_from_map, scribble_binary, smooth_map


@dataclass(frozen=True)
class HeadScoreConfig:
    metric: str = "selected"  # "selected" uses enabled guidance losses; "coverage" always cross-entropy.
    use_coverage: bool = False
    use_focal: bool = True
    use_moment: bool = True
    use_centerline: bool = True
    coverage_weight: float = 1.0
    focal_alpha: float = 0.25
    focal_beta: float = 2.0
    focal_weight: float = 3.0
    moment_alpha: float = 1.0
    moment_weight: float = 1.0
    centerline_weight: float = 1.0
    smooth_attn: bool = True
    smooth_sigma: float = 0.5
    smooth_kernel_size: int = 3


@torch.no_grad()
def score_head_maps(
    head_maps: torch.Tensor,
    target: torch.Tensor,
    grid_h: int,
    grid_w: int,
    config: HeadScoreConfig,
) -> torch.Tensor:
    """Return [B,H] alignment loss per attention head (lower is better).

    Args:
        head_maps: [B,heads,H*W], raw object-token attention probabilities.
        target: [B,H*W], normalized scribble distribution; propagation already applied.
        grid_h, grid_w: exact non-square or square output token grid.
        config: score selection and (for 'selected') the *same* coefficients as
            the differentiable guidance loss on the averaged/weighted map.

    Uses vectorized losses, avoiding 1 model forward/backward per head.
    """
    if head_maps.ndim != 3:
        raise ValueError("head_maps must have shape [B,heads,spatial_tokens]")
    b, heads, n = head_maps.shape
    if n != grid_h * grid_w or target.shape != (b, n):
        raise ValueError("Head maps and target must match the given image token grid")
    if config.metric not in {"coverage", "selected"}:
        raise ValueError("metric must be 'coverage' or 'selected'")

    attn = head_maps.float().reshape(b, heads, grid_h, grid_w)
    if config.metric == "selected" and config.smooth_attn:
        attn = smooth_map(attn, config.smooth_kernel_size, config.smooth_sigma)
    stroke = scribble_binary(target, grid_h, grid_w).float().to(attn.device)
    stroke_flat = stroke.reshape(b, -1)
    stroke_dist = stroke_flat / stroke_flat.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    # Shape [B,heads] for every term.
    scores = torch.zeros((b, heads), dtype=torch.float32, device=attn.device)
    if config.metric == "coverage" or config.use_coverage:
        spatial = attn.flatten(2).clamp_min(1e-12)
        spatial = spatial / spatial.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        coverage = -(stroke_dist[:, None, :] * spatial.log()).sum(dim=-1)
        if config.metric == "coverage":
            return coverage
        scores += config.coverage_weight * coverage

    if config.use_focal:
        target_4d = stroke.expand(b, heads, grid_h, grid_w)
        cross_entropy = F.binary_cross_entropy_with_logits(attn, target_4d, reduction="none")
        pt = torch.exp(-cross_entropy)
        alpha = config.focal_alpha * target_4d + (1 - config.focal_alpha) * (1 - target_4d)
        focal = ((1 - pt) ** config.focal_beta * alpha * cross_entropy).mean(dim=(-2, -1))
        scores += config.focal_weight * focal

    if config.use_moment:
        min_map = attn.amin(dim=(-1, -2), keepdim=True)
        max_map = attn.amax(dim=(-1, -2), keepdim=True)
        normalized = (attn - min_map) / (max_map - min_map + 1e-5)
        angle_attn, cx_attn, cy_attn = _moment_from_map(normalized)
        angle_scrib, cx_scrib, cy_scrib = _moment_from_map(stroke)
        cx_diff = cx_attn / grid_w - cx_scrib / grid_w
        cy_diff = cy_attn / grid_h - cy_scrib / grid_h
        centroid = torch.sqrt(cx_diff.square() + cy_diff.square())
        angle_diff = (angle_attn - angle_scrib).abs()
        angle_diff = torch.minimum(angle_diff, 2 * math.pi - angle_diff) / math.pi
        moment = centroid + config.moment_alpha * angle_diff.square()
        scores += config.moment_weight * moment

    if config.use_centerline:
        # Calculate scribble distance once for each image, not once per head.
        distances = _distance_map_to_scribble(stroke)
        probs = attn.clamp_min(1e-12)
        probs = probs / probs.sum(dim=(-1, -2), keepdim=True).clamp_min(1e-12)
        centerline = (probs * distances).sum(dim=(-1, -2))
        scores += config.centerline_weight * centerline

    return scores


class OnlineHeadWeighter:
    """Weights heads *across selected blocks*; updated once per guided step.

    EMA factor 1.0 uses the most recent improvement only; 0.5 averages new and
    previous weights. Uniform mixing 0 keeps pure softmax; >0 protects against
    putting almost all mass on a few heads when feedback is noisy.
    """

    def __init__(
        self,
        temperature: float = 0.01,
        ema: float = 0.5,
        uniform_mix: float = 0.1,
    ):
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Head temperature must be > 0")
        if not math.isfinite(ema) or not 0 < ema <= 1:
            raise ValueError("Head EMA must be within (0,1]")
        if not math.isfinite(uniform_mix) or not 0 <= uniform_mix <= 1:
            raise ValueError("Uniform mix must be within [0,1]")
        self.temperature = float(temperature)
        self.ema = float(ema)
        self.uniform_mix = float(uniform_mix)
        self.weights: torch.Tensor | None = None
        self.labels: list[str] = []

    def initialize(self, labels: list[str], device: torch.device) -> None:
        if not labels or len(set(labels)) != len(labels):
            raise ValueError("Expected unique nonempty attention head labels")
        if self.weights is not None:
            if self.labels != labels:
                raise RuntimeError("Head list changed mid-generation")
            return
        self.labels = labels
        self.weights = torch.full((len(labels),), 1 / len(labels), device=device, dtype=torch.float32)

    def aggregate(self, head_maps: torch.Tensor) -> torch.Tensor:
        """Weighted object attention [B,N] for the *actual gradient loss*."""
        if self.weights is None:
            raise RuntimeError("Initialize the weighter from the captured heads first")
        if head_maps.shape[1] != self.weights.numel():
            raise ValueError("Captured head count changed")
        # weights are treated as constants; gradients flow into head maps -> z_t.
        return (head_maps * self.weights.to(head_maps.device)[None, :, None]).sum(dim=1)

    @torch.no_grad()
    def update(self, before_losses: torch.Tensor, after_losses: torch.Tensor, step: int) -> dict:
        if self.weights is None:
            raise RuntimeError("Online head weighting must be initialized first")
        if before_losses.shape != after_losses.shape or before_losses.ndim != 2:
            raise ValueError("Per-head before/after losses must both have shape [B,heads]")
        if before_losses.shape[1] != self.weights.numel():
            raise ValueError("Per-head losses don't match head weight count")
        if not torch.isfinite(before_losses).all() or not torch.isfinite(after_losses).all():
            raise RuntimeError("Non-finite attention alignment scores: inspect head maps")

        delta = (before_losses.float() - after_losses.float()).mean(dim=0)
        proposed = torch.softmax(delta / self.temperature, dim=0)
        proposed = (1 - self.uniform_mix) * proposed + self.uniform_mix / self.weights.numel()
        next_weights = (1 - self.ema) * self.weights + self.ema * proposed
        self.weights = next_weights / next_weights.sum()

        order = torch.argsort(self.weights, descending=True)
        top = [{
            "head": self.labels[int(idx)],
            "weight": float(self.weights[idx].item()),
            "improvement": float(delta[idx].item()),
        } for idx in order[:min(8, self.weights.numel())].tolist()]
        entropy = -(self.weights * self.weights.clamp_min(1e-12).log()).sum()
        return {
            "step": int(step),
            "head_labels": self.labels,
            "improvement": delta.cpu().tolist(),
            "weights_next": self.weights.cpu().tolist(),
            "top_heads": top,
            "effective_heads": float(entropy.exp().item()),
        }
