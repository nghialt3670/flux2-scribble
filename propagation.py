"""ScribbleDiff-inspired propagation on a FLUX.2 Klein image-token grid.

The sparse strokes are expanded using *output-image -> output-image* joint
attention, normalized within the output-image tokens. At a given denoising step:

1. Average the self-attention maps of a few chosen transformer blocks.
2. Build a prototype attention distribution from the ORIGINAL scribble pixels.
3. Find unvisited 4-connected neighbors of the current expanded scribble.
4. Rank candidates by symmetric KL divergence to the scribble prototype.
5. Add up to top_k candidates with divergence below a user-set threshold.

Only the spatial guidance target changes. This routine does not modify the
transformer weights, the attention outputs, or the noisy image latent. It is an
adaptation of ScribbleDiff's region-growing principle, not an exact U-Net port.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _frontier_4(active: torch.Tensor) -> torch.Tensor:
    """4-connected unvisited neighbors for [B,H,W] boolean masks."""
    neighbors = torch.zeros_like(active)
    neighbors[:, 1:, :] |= active[:, :-1, :]
    neighbors[:, :-1, :] |= active[:, 1:, :]
    neighbors[:, :, 1:] |= active[:, :, :-1]
    neighbors[:, :, :-1] |= active[:, :, 1:]
    return neighbors & ~active


class ScribblePropagator:
    """Stateful single-object propagator; masks remain on the image-token grid.

    Args:
        seed_mask: [B,1,H,W] source scribble image mask. Downsampled via
            adaptive max-pooling to avoid losing thin brush strokes.
        grid_h, grid_w: Output token grid height and width.
        top_k: Maximum *new* cells added per propagation step.
        threshold: Maximum symmetric KL divergence for a candidate cell.
            This is NOT ScribbleDiff's original numerical threshold: we use
            normalized image-to-image probabilities, so its scale differs.
        max_area_ratio: Stop once this share of image tokens is occupied.
    """

    def __init__(
        self,
        seed_mask: torch.Tensor,
        grid_h: int,
        grid_w: int,
        *,
        top_k: int = 12,
        threshold: float = 0.2,
        max_area_ratio: float = 0.30,
    ):
        if seed_mask.ndim != 4 or seed_mask.shape[1] != 1:
            raise ValueError("seed_mask must have shape [B,1,H,W]")
        if grid_h < 1 or grid_w < 1:
            raise ValueError("Grid height and width must be positive")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if threshold <= 0:
            raise ValueError("threshold must be > 0")
        if not 0 < max_area_ratio <= 1:
            raise ValueError("max_area_ratio must be in (0, 1]")

        self.h, self.w = int(grid_h), int(grid_w)
        self.top_k = int(top_k)
        self.threshold = float(threshold)
        self.max_area_ratio = float(max_area_ratio)
        seed = F.adaptive_max_pool2d(seed_mask.float(), (self.h, self.w))[:, 0] > 0
        if not seed.any(dim=(1, 2)).all():
            raise ValueError("At least one scribble cell is required per example")
        self.seed = seed
        self.active = seed.clone()

    def target(self) -> torch.Tensor:
        """Normalized expanded scribble distribution [B,H*W]."""
        distribution = self.active.flatten(1).float()
        return distribution / distribution.sum(dim=-1, keepdim=True).clamp_min(1)

    @torch.no_grad()
    def propagate(self, self_attention: torch.Tensor) -> dict:
        """Expand toward image tokens with similar self-attention distributions.

        self_attention: [B,N,N], rows corresponding to output-image queries,
                        columns to output-image keys. No gradients needed.
        """
        b, n = self.seed.shape[0], self.h * self.w
        if self_attention.shape != (b, n, n):
            raise ValueError(f"self_attention must be [{b},{n},{n}]; got {tuple(self_attention.shape)}")

        a = self_attention.detach().float().clamp_min(1e-12)
        a = a / a.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        active = self.active.clone()
        newly_added = []
        min_distances = []

        for k in range(b):
            active_flat = active[k].flatten()
            original_flat = self.seed[k].flatten()
            room = max(0, int(self.max_area_ratio * n) - int(active_flat.sum().item()))
            candidates = _frontier_4(active[k:k+1]).flatten().nonzero(as_tuple=False).flatten()
            if not room or candidates.numel() == 0:
                newly_added.append(0)
                min_distances.append(None)
                continue

            # Anchor to the original stroke to avoid prototype drift.
            proto = a[k, original_flat].mean(dim=0).clamp_min(1e-12)
            proto = proto / proto.sum()
            neighbor = a[k].index_select(0, candidates).clamp_min(1e-12)

            # Symmetric KL: (KL(P||Q) + KL(Q||P))/2.
            log_p = proto.log().unsqueeze(0)
            log_q = neighbor.log()
            dist = 0.5 * (
                (neighbor * (log_q - log_p)).sum(dim=-1)
                + (proto.unsqueeze(0) * (log_p - log_q)).sum(dim=-1)
            )
            count = min(int(self.top_k), int(room), int(candidates.numel()))
            vals, local_idx = torch.topk(dist, count, largest=False)
            keep = vals < self.threshold
            selected = candidates[local_idx[keep]]
            active_flat[selected] = True
            newly_added.append(int(keep.sum().item()))
            min_distances.append(float(vals[0].item()) if vals.numel() else None)

        self.active = active
        return {
            "added": newly_added[0] if b == 1 else newly_added,
            "total": int(active.sum().item()) if b == 1 else active.flatten(1).sum(dim=1).tolist(),
            "min_kl": min_distances[0] if b == 1 else min_distances,
        }
