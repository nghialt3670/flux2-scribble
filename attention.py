"""Extract differentiable output-image -> object-text attention without modifying Klein.

Uses forward PRE-hooks on Flux2 attention modules; the original attention processors
still compute the model output. We reconstruct Q/K (including QK norms and RoPE)
only for guidance. Avoids copying the model's attention processor implementation.

Supported blocks: D0..D4 (double-stream) and S0..S19 (single-stream).
"""
from __future__ import annotations

import math
import re
from contextlib import contextmanager

import torch
from diffusers.models.embeddings import apply_rotary_emb


class AttentionCollector:
    def __init__(
        self,
        transformer,
        target_text_tokens: list[int],
        n_output_image_tokens: int,
        blocks: list[str],
        query_chunk: int = 64,
    ):
        self.transformer = transformer
        self.tokens = target_text_tokens
        self.n_output = n_output_image_tokens
        self.blocks = blocks
        self.query_chunk = query_chunk
        self.maps: dict[str, torch.Tensor] = {}
        if not self.tokens:
            raise ValueError("Target object text token selection is empty")

    @staticmethod
    def block_module(transformer, name):
        match = re.fullmatch(r"([DS])(\d+)", name.upper())
        if not match:
            raise ValueError(f"Invalid block {name!r}; use D0 or S3 format")
        kind, index = match.group(1), int(match.group(2))
        sequence = transformer.transformer_blocks if kind == "D" else transformer.single_transformer_blocks
        if index >= len(sequence):
            raise ValueError(f"Block {name} unavailable (only {len(sequence)} {kind} blocks)")
        return sequence[index].attn

    def _calculate(self, name: str, attn, kwargs):
        hidden = kwargs["hidden_states"]
        text_hidden = kwargs.get("encoder_hidden_states")
        rotary = kwargs.get("image_rotary_emb")
        head_dim = attn.head_dim
        if name.startswith("D"):
            if text_hidden is None:
                raise RuntimeError("Double-stream attention missing encoder_hidden_states")
            text_len = text_hidden.shape[1]
            if getattr(attn, "fused_projections", False):
                q_img, k_img, _ = attn.to_qkv(hidden).chunk(3, dim=-1)
                q_txt, k_txt, _ = attn.to_added_qkv(text_hidden).chunk(3, dim=-1)
            else:
                q_img, k_img = attn.to_q(hidden), attn.to_k(hidden)
                q_txt, k_txt = attn.add_q_proj(text_hidden), attn.add_k_proj(text_hidden)
            q_img = attn.norm_q(q_img.unflatten(-1, (-1, head_dim)))
            k_img = attn.norm_k(k_img.unflatten(-1, (-1, head_dim)))
            q_txt = attn.norm_added_q(q_txt.unflatten(-1, (-1, head_dim)))
            k_txt = attn.norm_added_k(k_txt.unflatten(-1, (-1, head_dim)))
            q = torch.cat([q_txt, q_img], dim=1)
            k = torch.cat([k_txt, k_img], dim=1)
        else:
            # Single-stream input is [text, output-image, reference-image] tokens.
            # Project only QKV, avoiding the fused MLP input columns.
            text_len = self.text_len
            qkv_width = 3 * attn.inner_dim
            layer = attn.to_qkv_mlp_proj
            bias = layer.bias[:qkv_width] if layer.bias is not None else None
            q, k, _ = torch.nn.functional.linear(hidden, layer.weight[:qkv_width], bias).chunk(3, dim=-1)
            q = attn.norm_q(q.unflatten(-1, (-1, head_dim)))
            k = attn.norm_k(k.unflatten(-1, (-1, head_dim)))

        if rotary is not None:
            q = apply_rotary_emb(q, rotary, sequence_dim=1)
            k = apply_rotary_emb(k, rotary, sequence_dim=1)

        q = q[:, text_len : text_len + self.n_output]
        if q.shape[1] != self.n_output:
            raise RuntimeError(f"Expected {self.n_output} output-image queries, got {q.shape[1]}")
        if max(self.tokens) >= text_len:
            raise RuntimeError(f"Target token index exceeds text sequence of length {text_len}")

        # Exact probability mass in target text tokens, normalized over ALL
        # text + output image + reference image keys (not text-only softmax).
        k = k.transpose(1, 2).float()  # [B,H,Lkeys,D]
        q = q.transpose(1, 2).float()  # [B,H,Lout,D]
        inds = torch.tensor(self.tokens, device=q.device, dtype=torch.long)
        chunks = []
        for pos in range(0, q.shape[2], self.query_chunk):
            logits = (q[:, :, pos : pos + self.query_chunk] @ k.transpose(-1, -2)) / math.sqrt(head_dim)
            numer = logits.index_select(-1, inds).logsumexp(dim=-1)
            denom = logits.logsumexp(dim=-1)
            # Average heads; keep gradients all the way to the noisy latent.
            chunks.append(torch.exp(numer - denom).mean(dim=1))
        self.maps[name] = torch.cat(chunks, dim=1)

    @contextmanager
    def capture(self, text_len: int):
        self.text_len = text_len
        self.maps = {}
        handles = []
        try:
            for name in self.blocks:
                module = self.block_module(self.transformer, name)

                def hook(attn, args, kwargs, _name=name):
                    if "hidden_states" not in kwargs:
                        raise RuntimeError("Klein attention hook expected keyword hidden_states; update attention.py")
                    self._calculate(_name, attn, kwargs)

                handles.append(module.register_forward_pre_hook(hook, with_kwargs=True))
            yield self
        finally:
            for h in handles:
                h.remove()

    def mean_map(self) -> torch.Tensor:
        absent = set(self.blocks) - set(self.maps)
        if absent:
            raise RuntimeError(f"Attention hooks not called: {sorted(absent)}")
        return torch.stack([self.maps[name] for name in self.blocks]).mean(dim=0)


def find_object_token_indices(tokenizer, prompt: str, object_phrase: str, max_length: int) -> list[int]:
    """Map the object phrase to Qwen chat-template token positions via character offsets."""
    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    start = formatted.lower().find(object_phrase.lower())
    if start < 0:
        raise ValueError(f"Object phrase {object_phrase!r} not found in prompt {prompt!r}")
    end = start + len(object_phrase)
    tok = tokenizer(
        formatted, max_length=max_length, padding="max_length", truncation=True,
        return_offsets_mapping=True,
    )
    hits = [i for i, (a, b) in enumerate(tok["offset_mapping"]) if a < end and b > start and b > a]
    if not hits:
        raise ValueError("Object phrase was truncated or not tokenized; shorten prompt or increase --max-text-tokens")
    return hits
