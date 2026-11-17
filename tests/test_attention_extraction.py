"""Test attention extraction without requiring Diffusers checkpoint downloads."""
import importlib
import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Install a tiny stub for Diffusers' RoPE function so we can test the pre-hook
# bookkeeping without needing the heavy model dependency in CI.
if 'diffusers' not in sys.modules:
    diffusers = types.ModuleType('diffusers')
    diffusers.__path__ = []
    models = types.ModuleType('diffusers.models')
    models.__path__ = []
    embedding = types.ModuleType('diffusers.models.embeddings')
    embedding.apply_rotary_emb = lambda x, emb, sequence_dim=1: x
    sys.modules.update({
        'diffusers': diffusers, 'diffusers.models': models,
        'diffusers.models.embeddings': embedding,
    })

from attention import AttentionCollector


class IdentityAttention(torch.nn.Module):
    def __init__(self, dim=8, heads=2):
        super().__init__()
        self.head_dim = dim // heads
        self.fused_projections = False
        self.to_q = torch.nn.Linear(dim, dim, bias=False)
        self.to_k = torch.nn.Linear(dim, dim, bias=False)
        self.add_q_proj = torch.nn.Linear(dim, dim, bias=False)
        self.add_k_proj = torch.nn.Linear(dim, dim, bias=False)
        self.norm_q = torch.nn.Identity()
        self.norm_k = torch.nn.Identity()
        self.norm_added_q = torch.nn.Identity()
        self.norm_added_k = torch.nn.Identity()

    def forward(self, *, hidden_states, encoder_hidden_states, image_rotary_emb=None):
        return hidden_states


class DoubleBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = IdentityAttention()


class FakeTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([DoubleBlock()])
        self.single_transformer_blocks = torch.nn.ModuleList([])


def test_joint_attention_maps_are_differentiable_and_split_image_only():
    torch.manual_seed(42)
    transformer = FakeTransformer()
    # hidden contains 4 output image tokens followed by 2 reference tokens.
    x = torch.randn(1, 6, 8, requires_grad=True)
    enc = torch.randn(1, 3, 8)
    extractor = AttentionCollector(transformer, [1], 4, ['D0'], self_blocks=['D0'], query_chunk=2)
    with extractor.capture(text_len=3, capture_self=True):
        transformer.transformer_blocks[0].attn(
            hidden_states=x, encoder_hidden_states=enc, image_rotary_emb=None
        )
    obj = extractor.mean_map()
    self_map = extractor.mean_self_map()
    assert obj.shape == (1, 4)
    assert self_map.shape == (1, 4, 4)
    assert torch.allclose(self_map.sum(dim=-1), torch.ones(1, 4), atol=1e-5)
    assert torch.isfinite(obj).all()
    assert (obj.sum()).backward() is None
    assert x.grad is not None and x.grad.abs().sum() > 0
    assert not self_map.requires_grad
