"""CPU-only tests for the self-attention scribble growth algorithm."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from propagation import ScribblePropagator, _frontier_4


def test_four_neighbors_no_wrap():
    m = torch.zeros(1, 3, 3, dtype=torch.bool)
    m[0, 0, 0] = True
    out = _frontier_4(m)
    assert set(map(tuple, out[0].nonzero().tolist())) == {(0, 1), (1, 0)}


def test_identical_pattern_neighbor_grows_first():
    seed = torch.zeros(1, 1, 3, 3)
    seed[0, 0, 1, 1] = 1
    # Every attention row is a valid probability distribution. Only the
    # neighbor directly above the original scribble has a matching pattern.
    a = torch.full((1, 9, 9), 0.001)
    a[:, :, 0] = 0.992
    a[:, 4, :] = 0.001
    a[:, 4, 3] = 0.992
    a[:, 1, :] = 0.001
    a[:, 1, 3] = 0.992
    prop = ScribblePropagator(seed, 3, 3, top_k=4, threshold=0.1, max_area_ratio=1.0)
    event = prop.propagate(a)
    assert event["added"] == 1
    assert prop.active[0, 0, 1].item()
    assert prop.active[0].sum().item() == 2
    assert torch.allclose(prop.target().sum(dim=1), torch.ones(1))


def test_never_overwrite_seed_and_respect_area_cap():
    seed = torch.zeros(1, 1, 4, 4)
    seed[0, 0, 1, 1] = 1
    a = torch.ones(1, 16, 16) / 16
    prop = ScribblePropagator(seed, 4, 4, top_k=16, threshold=0.1, max_area_ratio=0.25)
    for _ in range(10):
        prop.propagate(a)
    assert prop.active[0, 1, 1].item()
    assert prop.active.sum().item() == 4


def test_rejects_invalid_attention_size():
    seed = torch.ones(1, 1, 3, 3)
    prop = ScribblePropagator(seed, 3, 3)
    with pytest.raises(ValueError):
        prop.propagate(torch.ones(1, 3, 3))
