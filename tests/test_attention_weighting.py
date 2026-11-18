"""Adaptive weighting tests; require only torch, not a FLUX.2 checkpoint."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attention_weighting import HeadScoreConfig, OnlineHeadWeighter, score_head_maps
from guidance import compute_total_guidance_loss


def test_initial_head_average_and_gradient():
    torch.manual_seed(3)
    heads = torch.rand(1, 4, 12, requires_grad=True)
    adapter = OnlineHeadWeighter()
    adapter.initialize([f"D0:H{i:02d}" for i in range(4)], heads.device)
    actual = adapter.aggregate(heads)
    assert torch.allclose(actual, heads.mean(dim=1), atol=1e-7)
    actual.sum().backward()
    assert heads.grad is not None and heads.grad.abs().sum().item() > 0
    assert torch.allclose(heads.grad, torch.full_like(heads, 0.25))


def test_loss_improvement_increases_one_head_weight():
    # The first head moves attention toward the last pixel, the second worsens.
    initial = torch.tensor([[[0.8, 0.1, 0.1], [0.4, 0.3, 0.3]]])
    updated = torch.tensor([[[0.1, 0.1, 0.8], [0.8, 0.1, 0.1]]])
    target = torch.tensor([[0., 0., 1.]])
    config = HeadScoreConfig(metric="coverage")
    before = score_head_maps(initial, target, 1, 3, config)
    after = score_head_maps(updated, target, 1, 3, config)
    assert before[0, 0] > after[0, 0]
    assert before[0, 1] < after[0, 1]
    adapter = OnlineHeadWeighter(temperature=0.1, ema=1, uniform_mix=0)
    adapter.initialize(["D0:H00", "D0:H01"], initial.device)
    info = adapter.update(before, after, 0)
    assert adapter.weights[0] > adapter.weights[1]
    assert info["top_heads"][0]["head"] == "D0:H00"
    assert abs(adapter.weights.sum().item() - 1) < 1e-7


def test_selected_loss_score_matches_main_guidance_loss_rectangular():
    torch.manual_seed(2)
    for h, w in [(21, 32), (32, 21), (8, 8)]:
        attention = torch.rand(1, 2, h*w) * 0.2 + 0.02
        target = torch.zeros(1, h*w)
        target[0, (h//3)*w + w//4] = 1
        target[0, (h//2)*w + w//2] = 1
        target = target / target.sum(-1, keepdim=True)
        config = HeadScoreConfig(
            metric="selected", use_coverage=True,
            coverage_weight=0.7, focal_weight=2.3,
            moment_weight=1.2, centerline_weight=0.5,
        )
        scores = score_head_maps(attention, target, h, w, config)
        for idx in range(attention.shape[1]):
            expected, _, _ = compute_total_guidance_loss(
                attention[:, idx], target, h, w,
                use_coverage=config.use_coverage,
                use_focal=config.use_focal,
                use_moment=config.use_moment,
                use_centerline=config.use_centerline,
                coverage_weight=config.coverage_weight,
                focal_alpha=config.focal_alpha,
                focal_beta=config.focal_beta,
                focal_weight=config.focal_weight,
                moment_alpha=config.moment_alpha,
                moment_weight=config.moment_weight,
                centerline_weight=config.centerline_weight,
                smooth_attn=config.smooth_attn,
                sigma=config.smooth_sigma,
                kernel_size=config.smooth_kernel_size,
            )
            assert torch.allclose(scores[0, idx], expected.detach(), atol=2e-5, rtol=2e-5), (h, w, idx, scores[0, idx], expected)


def test_ema_and_uniform_fallback():
    adapter = OnlineHeadWeighter(temperature=0.01, ema=0.5, uniform_mix=0.2)
    adapter.initialize(["D0:H00", "D0:H01"], torch.device('cpu'))
    before = torch.tensor([[10., 10.]])
    after = torch.tensor([[9., 11.]])
    adapter.update(before, after, 0)
    assert adapter.weights[0] > adapter.weights[1]
    assert adapter.weights.min() > 0
    adapter_uniform = OnlineHeadWeighter(uniform_mix=1, ema=1)
    adapter_uniform.initialize(["D0:H00", "D0:H01"], torch.device('cpu'))
    adapter_uniform.update(before, after, 0)
    assert torch.allclose(adapter_uniform.weights, torch.tensor([0.5, 0.5]))


def test_invalid_parameters_or_shapes():
    for kw in ({"temperature": 0}, {"ema": 0}, {"ema": 1.1}, {"uniform_mix": -0.1}):
        with pytest.raises(ValueError):
            OnlineHeadWeighter(**kw)
    with pytest.raises(ValueError):
        score_head_maps(torch.rand(1, 2, 16), torch.ones(1, 16), 3, 3, HeadScoreConfig())
    with pytest.raises(RuntimeError):
        OnlineHeadWeighter().aggregate(torch.rand(1, 2, 16))
