"""Regression: rectangular attention grids must work with meaningful x/y moments."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from guidance import _moment_from_map, compute_total_guidance_loss


def test_rectangular_centroid():
    a = torch.zeros(1, 1, 21, 32)
    a[0, 0, 7, 19] = 1
    angle, x, y = _moment_from_map(a)
    assert abs(x.item() - 19) < 1e-2
    assert abs(y.item() - 7) < 1e-2


def test_rectangular_gradient():
    torch.manual_seed(4)
    for h, w in [(21, 32), (32, 21), (32, 32)]:
        attn = torch.rand((1, h * w), requires_grad=True)
        target = torch.zeros((1, h * w))
        target[0, (h // 2) * w + w // 2] = 1
        loss, terms, _ = compute_total_guidance_loss(attn, target, h, w)
        grad = torch.autograd.grad(loss, attn)[0]
        assert torch.isfinite(loss)
        assert torch.isfinite(grad).all()
        assert grad.abs().sum() > 0
        assert set(terms) == {'focal', 'moment', 'centerline', 'total'}
