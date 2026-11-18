"""CPU tests: individual head control, gradient, exported statistics and maps."""
import csv
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from attention_weighting import manual_blocks, manual_weights, parse_manual_heads, aggregate_head_maps, HeadScoreConfig
from head_diagnostics import HeadDiagnostics


def test_manual_parser_and_blocks():
    spec = 'D2:H3, D2:H07=2; S4:H1=0.5'
    assert parse_manual_heads(spec) == {'D2:H03': 1, 'D2:H07': 2, 'S4:H01': .5}
    assert manual_blocks(spec) == ['D2', 'S4']
    for wrong in ['', 'D2:H3,D2:H03', 'D2:H3=0', 'D2:H3=-1', 'X2:H1', 'D2:H3=nan']:
        with pytest.raises(ValueError):
            parse_manual_heads(wrong)


def test_manual_selection_controls_gradient():
    labels = ['D2:H00', 'D2:H01', 'D2:H02', 'S4:H00']
    weights = manual_weights(labels, 'D2:H1=3,S4:H0=1', 'cpu')
    assert torch.allclose(weights, torch.tensor([0., .75, 0., .25]))
    source = torch.arange(16., requires_grad=True).reshape(1, 4, 4)
    combined = aggregate_head_maps(source, weights)
    assert torch.allclose(combined, source[:, 1] * .75 + source[:, 3] * .25)
    grad = torch.autograd.grad(combined.sum(), source)[0]
    assert torch.allclose(grad[0, :, 0], weights)
    with pytest.raises(ValueError, match='unavailable'):
        manual_weights(labels, 'D2:H99', 'cpu')


def test_diagnostics_two_blocks_rectangular(tmp_path):
    h, w = 21, 32
    img = Image.new('RGB', (w*8, h*8), '#777777')
    diag = HeadDiagnostics(tmp_path, img, h, w, interval=2, score_config=HeadScoreConfig(metric='coverage'))
    labels = ['D2:H00', 'D2:H01', 'S4:H00']
    target = torch.zeros(1, h*w)
    target[0, 8*w+9:8*w+12] = 1
    target /= target.sum()
    weights = manual_weights(labels, 'D2:H1=2,S4:H0=1', 'cpu')
    maps = torch.rand(1, 3, h*w) * .05 + .001
    diag.record(step=0, timestep=1000., labels=labels, maps=maps, target=target, weights=weights)
    assert (tmp_path/'attention_heads/head_maps/step_000/D2_H00.png').exists()
    assert (tmp_path/'attention_heads/head_maps/step_000/S4_H00.png').exists()
    assert (tmp_path/'attention_heads/contact_sheets/step_000_D2.png').exists()
    assert (tmp_path/'attention_heads/contact_sheets/step_000_S4.png').exists()
    diag.record_after(0, torch.tensor([[1., 2., 3.]]))
    diag.record(step=1, timestep=800., labels=labels, maps=maps, target=target,
                weights=weights, is_last_guided=True)
    diag.finalize()
    root = tmp_path/'attention_heads'
    assert (root/'head_maps/step_001/D2_H01.png').exists()
    assert (root/'raw_maps/step_000.npz').exists()
    assert (root/'raw_maps/step_001.npz').exists()
    import numpy as np
    with np.load(root/'raw_maps/step_001.npz') as data:
        assert data['maps'].shape == (3, h, w)
        assert list(data['head_labels']) == labels
    with (root/'head_steps.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 6
    assert rows[0]['head'] == 'D2:H00'
    assert rows[0]['selected'] == '0'
    assert rows[0]['loss_after'] != ''
    assert rows[3]['loss_after'] == ''
    with (root/'head_summary.csv').open() as f:
        summaries = list(csv.DictReader(f))
    assert len(summaries) == 3
    assert diag.last_atlas.exists()
