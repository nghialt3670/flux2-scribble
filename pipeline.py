"""Training-free per-timestep scribble guidance using Diffusers' native Klein helpers.

This is an explicit, batch-size-1 sampling loop. The conditional transformer is
run once for the gradient, then again (without grad) for the actual velocity.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu, retrieve_timesteps
from diffusers.utils.torch_utils import get_module_execution_device

from attention import AttentionCollector, find_object_token_indices
from guidance import attention_coverage_loss, normalized_latent_update, scribble_target


def _forward(
    pipe,
    latents: torch.Tensor,
    step_t: torch.Tensor,
    prompt_embeds: torch.Tensor,
    txt_ids: torch.Tensor,
    latent_ids: torch.Tensor,
    ref_latents: torch.Tensor | None,
    ref_ids: torch.Tensor | None,
    cache_branch: str,
):
    model_dtype = pipe.transformer.dtype
    model_input = latents.to(dtype=model_dtype)
    ids = latent_ids
    if ref_latents is not None:
        model_input = torch.cat([model_input, ref_latents.to(dtype=model_dtype)], dim=1)
        ids = torch.cat([latent_ids, ref_ids], dim=1)

    # Same normalized timestep convention as the official Klein pipeline.
    timestep = step_t.expand(latents.shape[0]).to(dtype=model_dtype) / 1000
    with pipe.transformer.cache_context(cache_branch):
        result = pipe.transformer(
            hidden_states=model_input,
            timestep=timestep,
            guidance=None,
            encoder_hidden_states=prompt_embeds,
            txt_ids=txt_ids,
            img_ids=ids,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]
    return result[:, : latents.shape[1]]


def generate_guided(
    pipe,
    image,
    scribble_mask: torch.Tensor,
    prompt: str,
    object_phrase: str,
    *,
    num_steps: int = 30,
    guide_steps: int = 10,
    guidance_strength: float = 0.01,
    cfg_scale: float = 4.0,
    blocks: list[str] | None = None,
    seed: int = 42,
    max_text_tokens: int = 256,
    attention_query_chunk: int = 64,
    debug_dir: str | Path | None = None,
):
    """Return (PIL output, stats). Image and scribble are already resized together.

    Uses the same reference-image token conditioning as Flux2KleinPipeline. It is
    not a masked inpainting implementation: unedited pixels are not locked.
    """
    if num_steps < 1 or guide_steps < 0 or guide_steps > num_steps:
        raise ValueError("Require steps >= 1 and 0 <= guide_steps <= steps")
    if cfg_scale < 1:
        raise ValueError("CFG scale must be >= 1")
    if guidance_strength < 0:
        raise ValueError("Guidance strength must be nonnegative")
    if blocks is None:
        blocks = [f"D{i}" for i in range(len(pipe.transformer.transformer_blocks))]
    if not blocks and guide_steps:
        raise ValueError("At least one attention block is required")

    width, height = image.size
    multiple = pipe.vae_scale_factor * 2
    if width % multiple or height % multiple:
        raise ValueError(f"Image dimensions must be multiples of {multiple}; got {width}x{height}")

    if not hasattr(pipe.transformer, "cache_context"):
        raise RuntimeError("Please use the current Diffusers Git version (missing cache_context)")

    # No model weight updates.
    pipe.transformer.requires_grad_(False)
    device = pipe._execution_device
    model_dtype = pipe.transformer.dtype
    generator = torch.Generator(device="cpu").manual_seed(seed)

    with torch.no_grad():
        prompt_embeds, text_ids = pipe.encode_prompt(
            prompt=prompt,
            prompt_embeds=None,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=max_text_tokens,
            text_encoder_out_layers=(9, 18, 27),
        )
        negative_embeds = negative_ids = None
        if cfg_scale > 1:
            negative_embeds, negative_ids = pipe.encode_prompt(
                prompt="",
                prompt_embeds=None,
                device=device,
                num_images_per_prompt=1,
                max_sequence_length=max_text_tokens,
                text_encoder_out_layers=(9, 18, 27),
            )
        token_ids = find_object_token_indices(
            pipe.tokenizer, prompt, object_phrase, max_length=max_text_tokens
        )
        print(f"Object tokens for {object_phrase!r}: {token_ids}", flush=True)

        latents, latent_ids = pipe.prepare_latents(
            batch_size=1,
            num_latents_channels=pipe.transformer.config.in_channels // 4,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
            latents=None,
        )

        # Matching the native Klein editing/reference-image preparation path.
        processed = pipe.image_processor.preprocess(
            image, height=height, width=width, resize_mode="crop"
        )
        ref_latents, ref_ids = pipe.prepare_image_latents(
            images=[processed],
            batch_size=1,
            generator=generator,
            device=device,
            dtype=pipe.vae.dtype,
        )
        ref_latents = ref_latents.to(dtype=model_dtype)

        sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps)
        if getattr(pipe.scheduler.config, "use_flow_sigmas", False):
            sigmas = None
        mu = compute_empirical_mu(latents.shape[1], num_steps)
        timesteps, _ = retrieve_timesteps(
            pipe.scheduler, num_steps, device, sigmas=sigmas, mu=mu
        )
        if hasattr(pipe.scheduler, "set_begin_index"):
            pipe.scheduler.set_begin_index(0)

    grid_h = height // multiple
    grid_w = width // multiple
    if grid_h * grid_w != latents.shape[1]:
        raise RuntimeError("Packed output-token count does not match spatial grid")
    target = scribble_target(scribble_mask.to(device), grid_h, grid_w)
    collector = AttentionCollector(
        pipe.transformer, token_ids, latents.shape[1], blocks,
        query_chunk=attention_query_chunk,
    )
    saved_attention = None
    log = []

    for i, t in enumerate(timesteps):
        if i < guide_steps and guidance_strength > 0:
            with torch.enable_grad():
                # Keep a FP32 leaf even when Klein runs with BF16 weights.
                x = latents.detach().float().requires_grad_(True)
                with collector.capture(text_len=prompt_embeds.shape[1]):
                    # Velocity output is intentionally discarded; the backward
                    # pass uses only the differentiable attention maps.
                    _ = _forward(
                        pipe, x, t, prompt_embeds, text_ids,
                        latent_ids, ref_latents, ref_ids, "cond"
                    )
                attention = collector.mean_map()
                loss = attention_coverage_loss(attention, target)
                grad = torch.autograd.grad(loss, x)[0]
                x_new, grad_rms = normalized_latent_update(x, grad, guidance_strength)
                latents = x_new.to(dtype=latents.dtype)
                saved_attention = attention.detach()[0].float().cpu().reshape(grid_h, grid_w)
                loss_value = loss.item()
                del x, attention, loss, grad, x_new
                collector.maps.clear()
        else:
            loss_value, grad_rms = None, None

        # Actual Euler flow step on the UPDATED latent (both CFG branches).
        with torch.no_grad():
            prediction = _forward(
                pipe, latents, t, prompt_embeds, text_ids,
                latent_ids, ref_latents, ref_ids, "cond"
            )
            if cfg_scale > 1:
                negative_prediction = _forward(
                    pipe, latents, t, negative_embeds, negative_ids,
                    latent_ids, ref_latents, ref_ids, "uncond"
                )
                prediction = negative_prediction + cfg_scale * (prediction - negative_prediction)
            latents = pipe.scheduler.step(prediction, t, latents, return_dict=False)[0]
        log.append({"step": i, "timestep": float(t.item()), "loss": loss_value, "grad_rms": grad_rms})
        if loss_value is not None:
            print(f"[{i + 1:02d}/{len(timesteps)}] loss={loss_value:.5f} grad_rms={grad_rms:.3e}", flush=True)
        else:
            print(f"[{i + 1:02d}/{len(timesteps)}] normal denoising", flush=True)

    # Mirror Klein's VAE unpack/BN-unnormalization and decoding logic.
    with torch.no_grad():
        latent_h = 2 * (int(height) // multiple)
        latent_w = 2 * (int(width) // multiple)
        latents = pipe._unpack_latents_with_ids(
            latents, latent_ids, latent_h // 2, latent_w // 2
        )
        bn_mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        bn_std = torch.sqrt(
            pipe.vae.bn.running_var.view(1, -1, 1, 1).to(latents.device, latents.dtype)
            + pipe.vae.config.batch_norm_eps
        )
        latents = latents * bn_std + bn_mean
        latents = pipe._unpatchify_latents(latents)
        latents = latents.to(device=get_module_execution_device(pipe.vae), dtype=pipe.vae.dtype)
        pixels = pipe.vae.decode(latents, return_dict=False)[0]
        output = pipe.image_processor.postprocess(pixels, output_type="pil")[0]

    if debug_dir is not None:
        dest = Path(debug_dir)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "guidance_log.json").write_text(json.dumps(log, indent=2))
        if saved_attention is not None:
            np.save(dest / "attention_last_guided.npy", saved_attention.numpy())
    return output, {"log": log, "last_attention": saved_attention, "target_grid": target.detach().cpu()}
