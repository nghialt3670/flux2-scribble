"""ScribbleDiff-style losses plus a practical centerline loss for FLUX.2 Klein.

This module keeps the original simple utilities, but now exposes:
- ScribbleDiff focal loss
- ScribbleDiff moment loss (centroid + angle)
- Centerline distance loss (practical differentiable surrogate)

Important note:
ScribbleDiff's released code computes the moment loss on attention inside an
object mask. In this project we do not have a separate object mask at inference
-time, so we apply the same formula to the aggregated object-token attention
map itself. This preserves the loss structure while adapting it to our setting.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def scribble_target(mask: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Convert [B,1,H,W] stroke mask to a distribution on the image-token grid."""
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError("Expected scribble mask [B,1,H,W]")
    target = F.adaptive_max_pool2d(mask.float(), (h, w)).flatten(1)
    if not (target.sum(dim=-1) > 0).all():
        raise ValueError("The scribble contains no visible pixels after processing")
    return target / target.sum(dim=-1, keepdim=True)


def scribble_binary(mask_or_target: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Return a binary scribble grid [B,1,h,w] from an image mask or flat target."""
    if mask_or_target.ndim == 4:
        return (F.adaptive_max_pool2d(mask_or_target.float(), (h, w)) > 0).float()
    if mask_or_target.ndim == 2:
        if mask_or_target.shape[1] != h * w:
            raise ValueError("Flat target does not match target grid size")
        return (mask_or_target.reshape(mask_or_target.shape[0], 1, h, w) > 0).float()
    raise ValueError("Expected [B,1,H,W] mask or [B,H*W] target")


def _gaussian_kernel(kernel_size: int = 3, sigma: float = 0.5, device=None, dtype=None) -> torch.Tensor:
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(xx.square() + yy.square()) / (2 * sigma * sigma))
    kernel = kernel / kernel.sum().clamp_min(1e-12)
    return kernel


def smooth_map(x: torch.Tensor, kernel_size: int = 3, sigma: float = 0.5) -> torch.Tensor:
    """Depthwise Gaussian smoothing for [B,C,H,W]."""
    if kernel_size <= 1:
        return x
    kernel = _gaussian_kernel(kernel_size, sigma, device=x.device, dtype=x.dtype)
    kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(x.shape[1], 1, 1, 1)
    pad = kernel_size // 2
    x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    return F.conv2d(x, kernel, groups=x.shape[1])


def attention_coverage_loss(attn: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Kept for compatibility. Cross entropy between stroke distribution and attention."""
    if attn.shape != target.shape:
        raise ValueError(f"Attention shape {attn.shape} != scribble shape {target.shape}")
    attn = attn.float().clamp_min(1e-12)
    attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return -(target.float() * attn.log()).sum(dim=-1).mean()


def _moment_from_map(attn_obj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ScribbleDiff-style moment extraction.

    attn_obj: [B,1,H,W] or [B,C,H,W]
    returns angle, centroid_x, centroid_y with shape [B,C]
    """
    batch_size, obj_num, h, w = attn_obj.shape
    x, y = torch.meshgrid(
        torch.arange(0, w, device=attn_obj.device, dtype=attn_obj.dtype),
        torch.arange(0, h, device=attn_obj.device, dtype=attn_obj.dtype),
        indexing="ij",
    )
    x = x.expand(batch_size, obj_num, h, w)
    y = y.expand(batch_size, obj_num, h, w)

    m00 = torch.sum(attn_obj, dim=(-1, -2))
    m10 = torch.sum(x * attn_obj, dim=(-1, -2))
    m01 = torch.sum(y * attn_obj, dim=(-1, -2))
    eps = 1e-5
    x_avg = m10 / (m00 + eps)
    y_avg = m01 / (m00 + eps)
    mu20 = torch.sum((x - x_avg.unsqueeze(-1).unsqueeze(-1)) ** 2 * attn_obj, dim=(-1, -2))
    mu02 = torch.sum((y - y_avg.unsqueeze(-1).unsqueeze(-1)) ** 2 * attn_obj, dim=(-1, -2))
    mu11 = torch.sum((x - x_avg.unsqueeze(-1).unsqueeze(-1)) * (y - y_avg.unsqueeze(-1).unsqueeze(-1)) * attn_obj, dim=(-1, -2))
    angle = 0.5 * torch.atan2(2 * mu11, mu20 - mu02)
    angle[torch.isnan(angle)] = 0.0
    angle = torch.where(angle < 0, angle + math.pi, angle)
    return angle, x_avg, y_avg


def compute_focal_loss(attn_obj: torch.Tensor, scribble: torch.Tensor, alpha: float = 0.25, beta: float = 2.0) -> torch.Tensor:
    """Exact focal-loss formula used in ScribbleDiff's released code.

    attn_obj: [B,1,H,W] aggregated object attention map.
    scribble: [B,1,H,W] binary scribble mask.
    """
    target = scribble.float()
    cross_entropy_loss = torch.nn.BCEWithLogitsLoss(reduction="none")(attn_obj, target)
    pt = torch.exp(-cross_entropy_loss)
    at = alpha * target + (1.0 - alpha) * (1.0 - target)
    focal_loss = (1 - pt) ** beta * at * cross_entropy_loss
    return focal_loss.mean()


def compute_moment_loss(attn_obj: torch.Tensor, scribble: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """ScribbleDiff-style moment loss (centroid + squared angle loss).

    Adaptation for this project: because we do not have a separate object mask at
    inference time, we min-max normalize the aggregated attention map directly.
    """
    eps = 1e-5
    # Min-max normalize attention map before extracting moments.
    attn_min = attn_obj.amin(dim=(-1, -2), keepdim=True)
    attn_max = attn_obj.amax(dim=(-1, -2), keepdim=True)
    norm_attn = (attn_obj - attn_min) / (attn_max - attn_min + eps)

    attn_angle, attn_cx, attn_cy = _moment_from_map(norm_attn)
    scrib_angle, scrib_cx, scrib_cy = _moment_from_map(scribble.float())

    r = attn_obj.shape[-1]
    attn_cx = attn_cx / r
    attn_cy = attn_cy / r
    scrib_cx = scrib_cx / r
    scrib_cy = scrib_cy / r

    centroid_loss = torch.sqrt((attn_cx - scrib_cx) ** 2 + (attn_cy - scrib_cy) ** 2)
    angle_loss = torch.abs(attn_angle - scrib_angle)
    angle_loss = torch.min(angle_loss, 2 * math.pi - angle_loss) / math.pi
    angle_loss_squared = angle_loss ** 2
    return (centroid_loss + alpha * angle_loss_squared).mean()


def _distance_map_to_scribble(binary_scribble: torch.Tensor) -> torch.Tensor:
    """Brute-force distance transform on the small image-token grid.

    binary_scribble: [B,1,H,W] with values in {0,1}
    returns normalized distance map [B,1,H,W] in [0,1]
    """
    b, _, h, w = binary_scribble.shape
    ys, xs = torch.meshgrid(
        torch.arange(h, device=binary_scribble.device, dtype=torch.float32),
        torch.arange(w, device=binary_scribble.device, dtype=torch.float32),
        indexing="ij",
    )
    coords = torch.stack([ys, xs], dim=-1).reshape(-1, 2)
    outputs = []
    diag = math.sqrt(max(h - 1, 1) ** 2 + max(w - 1, 1) ** 2)
    for k in range(b):
        points = (binary_scribble[k, 0] > 0).nonzero(as_tuple=False).float()
        if points.numel() == 0:
            outputs.append(torch.ones((1, h, w), device=binary_scribble.device))
            continue
        distances = torch.cdist(coords.unsqueeze(0), points.unsqueeze(0))[0]
        nearest = distances.min(dim=1).values.reshape(1, h, w)
        outputs.append((nearest / max(diag, 1e-6)).clamp(0, 1))
    return torch.stack(outputs, dim=0)


def compute_centerline_loss(attn_obj: torch.Tensor, scribble: torch.Tensor) -> torch.Tensor:
    """Practical centerline loss.

    The user scribble is treated as the target centerline. We build a distance
    map to that centerline and penalize expected distance under the normalized
    attention distribution.
    """
    probs = attn_obj.float().clamp_min(1e-12)
    probs = probs / probs.sum(dim=(-1, -2), keepdim=True).clamp_min(1e-12)
    binary = (scribble > 0).float()
    dist = _distance_map_to_scribble(binary)
    return (probs * dist).sum(dim=(-1, -2)).mean()


def compute_total_guidance_loss(
    attn_flat: torch.Tensor,
    target: torch.Tensor,
    grid_h: int,
    grid_w: int,
    *,
    use_focal: bool = True,
    use_moment: bool = True,
    use_centerline: bool = True,
    focal_alpha: float = 0.25,
    focal_beta: float = 2.0,
    focal_weight: float = 3.0,
    moment_alpha: float = 1.0,
    moment_weight: float = 1.0,
    centerline_weight: float = 1.0,
    smooth_attn: bool = True,
    sigma: float = 0.5,
    kernel_size: int = 3,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    """Compute the full guidance loss from the aggregated attention map.

    Returns:
        total_loss,
        component dict,
        normalized 2D attention map [B,1,H,W]
    """
    b, n = attn_flat.shape
    if n != grid_h * grid_w:
        raise ValueError("Attention length does not match grid size")

    attn_map = attn_flat.reshape(b, 1, grid_h, grid_w).float()
    scribble = scribble_binary(target, grid_h, grid_w).to(attn_map.device, attn_map.dtype)
    if smooth_attn:
        attn_map = smooth_map(attn_map, kernel_size=kernel_size, sigma=sigma)

    components: dict[str, torch.Tensor] = {}
    total = torch.zeros((), device=attn_map.device, dtype=attn_map.dtype)

    if use_focal:
        focal = compute_focal_loss(attn_map, scribble, alpha=focal_alpha, beta=focal_beta)
        components["focal"] = focal
        total = total + focal_weight * focal

    if use_moment:
        moment = compute_moment_loss(attn_map, scribble, alpha=moment_alpha)
        components["moment"] = moment
        total = total + moment_weight * moment

    if use_centerline:
        centerline = compute_centerline_loss(attn_map, scribble)
        components["centerline"] = centerline
        total = total + centerline_weight * centerline

    components["total"] = total
    return total, components, attn_map


def normalized_latent_update(
    latent: torch.Tensor, grad: torch.Tensor, strength: float
) -> tuple[torch.Tensor, float]:
    """RMS-scaled gradient descent on latent; returns detached latent and gradient RMS."""
    rms = grad.float().square().mean().sqrt().clamp_min(1e-8)
    updated = (latent.float() - strength * grad.float() / rms).detach()
    return updated, rms.item()
