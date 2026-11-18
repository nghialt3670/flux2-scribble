"""Gradio drawing UI for FLUX.2 Klein Scribble Guidance."""
from __future__ import annotations

import argparse
import functools
from pathlib import Path
import time
import uuid

import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageOps

from input_utils import prepare_inputs
from visualize import save_summary

MODEL = "black-forest-labs/FLUX.2-klein-base-4B"


def editor_to_images(editor: dict | None) -> tuple[Image.Image, Image.Image]:
    if not editor or editor.get("background") is None:
        raise ValueError("Upload an image in the drawing canvas first")
    background = editor["background"]
    background = background if isinstance(background, Image.Image) else Image.fromarray(np.asarray(background))
    background = ImageOps.exif_transpose(background).convert("RGB")
    w, h = background.size
    stroke = np.zeros((h, w), dtype=np.uint8)
    for layer in editor.get("layers") or []:
        if layer is None:
            continue
        layer = layer if isinstance(layer, Image.Image) else Image.fromarray(np.asarray(layer))
        if layer.size != (w, h):
            raise ValueError(f"Canvas layer {layer.size} does not match image {(w, h)}")
        if "A" not in layer.getbands():
            raise ValueError("Drawing layer has no transparency; use the ImageEditor brush")
        alpha = np.asarray(layer.getchannel("A"), dtype=np.uint8)
        stroke = np.maximum(stroke, alpha)
    if not stroke.any():
        raise ValueError("Draw a red scribble on the image before generating")
    return background, Image.fromarray(stroke, "L")


def preview_stroke(editor: dict | None):
    try:
        _, mask = editor_to_images(editor)
        return mask
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc


def _device_and_dtype(selection: str, dtype_name: str):
    device = selection
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = (torch.float32 if device == "cpu" else torch.bfloat16) if dtype_name == "auto" else (
        torch.bfloat16 if dtype_name == "bf16" else torch.float32
    )
    return device, dtype


@functools.lru_cache(maxsize=8)
def _load_model(mode: str, device_selection: str, dtype_name: str, cpu_offload: bool):
    from diffusers import Flux2KleinInpaintPipeline, Flux2KleinPipeline

    device, dtype = _device_and_dtype(device_selection, dtype_name)
    pipe_cls = Flux2KleinPipeline if mode == "base" else Flux2KleinInpaintPipeline
    print(f"Loading {MODEL} in {mode} mode on {device}, dtype={dtype}", flush=True)
    pipe = pipe_cls.from_pretrained(MODEL, torch_dtype=dtype)
    if cpu_offload:
        if device != "cuda":
            raise ValueError("CPU offload is supported here only with the CUDA backend")
        pipe.enable_model_cpu_offload(device=device)
    else:
        pipe.to(device)
    pipe.transformer.eval()
    return pipe


def generate(
    editor, prompt, object_phrase, max_side, steps, guide_steps, strength,
    cfg, blocks_string, seed, device, dtype, cpu_offload,
    mode, inpaint_strength,
    attention_weighting, head_score_metric, head_temperature, head_ema, head_uniform_mix,
    use_coverage, use_focal, use_moment, use_centerline,
    coverage_weight, focal_alpha, focal_beta, focal_weight,
    moment_alpha, moment_weight, centerline_weight,
    propagate, prop_start, prop_end, prop_blocks_string, prop_top_k,
    prop_threshold, prop_max_area,
    progress=gr.Progress(),
):
    try:
        if not prompt or not prompt.strip():
            raise ValueError("Enter an editing prompt")
        if not object_phrase or object_phrase.lower() not in prompt.lower():
            raise ValueError("Object phrase must appear in the editing prompt")
        if int(guide_steps) > int(steps):
            raise ValueError("Guided steps cannot exceed total steps")
        blocks = [s.strip().upper() for s in blocks_string.split(",") if s.strip()]
        if not blocks or len(blocks) != len(set(blocks)):
            raise ValueError("Choose unique blocks, e.g. D0,D1,D2,D3,D4")
        prop_blocks = [s.strip().upper() for s in prop_blocks_string.split(",") if s.strip()]
        if propagate and (not prop_blocks or len(prop_blocks) != len(set(prop_blocks))):
            raise ValueError("Propagation blocks must be unique, e.g. D2,D3")

        original, stroke = editor_to_images(editor)
        image, mask, mask_tensor, original_size = prepare_inputs(original, stroke, int(max_side))
        progress(0.1, desc="Loading FLUX.2 Klein")
        pipe = _load_model(mode, device, dtype, bool(cpu_offload))
        from pipeline import generate_guided

        destination = Path("outputs") / f"ui_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        destination.mkdir(parents=True, exist_ok=True)
        image.save(destination / "input_processed.png")
        mask.save(destination / "scribble_processed.png")
        progress(0.25, desc="Guided denoising")
        result, info = generate_guided(
            pipe, image, mask_tensor, prompt.strip(), object_phrase.strip(),
            num_steps=int(steps), guide_steps=int(guide_steps),
            guidance_strength=float(strength), cfg_scale=float(cfg),
            blocks=blocks, seed=int(seed), debug_dir=destination,
            propagate=bool(propagate), prop_start=int(prop_start), prop_end=int(prop_end),
            prop_blocks=prop_blocks, prop_top_k=int(prop_top_k),
            prop_threshold=float(prop_threshold), prop_max_area=float(prop_max_area),
            edit_mode=str(mode), inpaint_strength=float(inpaint_strength),
            attention_weighting=str(attention_weighting),
            head_score_metric=str(head_score_metric),
            head_temperature=float(head_temperature),
            head_ema=float(head_ema),
            head_uniform_mix=float(head_uniform_mix),
            use_coverage=bool(use_coverage), use_focal=bool(use_focal), use_moment=bool(use_moment), use_centerline=bool(use_centerline),
            coverage_weight=float(coverage_weight),
            focal_alpha=float(focal_alpha), focal_beta=float(focal_beta), focal_weight=float(focal_weight),
            moment_alpha=float(moment_alpha), moment_weight=float(moment_weight),
            centerline_weight=float(centerline_weight),
        )
        result.save(destination / "result_native.png")
        if result.size != original_size:
            result = result.resize(original_size, Image.Resampling.LANCZOS)
        result.save(destination / "result.png")
        save_summary(image, mask, result, info["last_attention"], destination / "summary.png")
        progress(1.0, desc="Complete")

        lines = [
            f"Saved to {destination}",
            f"Mode: {mode}",
            f"Native inference resolution: {image.width}x{image.height}",
        ]
        guided = [e for e in info["log"] if e["loss"] is not None]
        if guided:
            last = guided[-1]
            lines.append(f"Guided steps: {len(guided)}; last total loss: {last['loss']:.5f}")
            if last.get("components"):
                c = last["components"]
                lines.append(
                    f"Last components — coverage: {c.get('coverage', 0.0):.5f}, focal: {c.get('focal', 0.0):.5f}, "
                    f"moment: {c.get('moment', 0.0):.5f}, centerline: {c.get('centerline', 0.0):.5f}"
                )
        else:
            lines.append("Guidance disabled")
        if info["weight_history"]:
            most_recent = info["weight_history"][-1]
            top = most_recent["top_heads"][:3]
            lines.append(
                "Adaptive heads (top 3): " + ", ".join(
                    f"{h['head']}={h['weight']:.3f}" for h in top
                )
            )
            lines.append(f"Effective head count: {most_recent['effective_heads']:.1f}")

        propagated = (
            Image.open(destination / "scribble_propagated.png").copy()
            if bool(propagate) and (destination / "scribble_propagated.png").exists()
            else mask
        )
        if bool(propagate):
            expansions = [e["propagation"]["added"] for e in info["log"] if e["propagation"]]
            lines.append(f"Propagation: {len(expansions)} iterations, {sum(expansions)} new grid cells")
        return result, mask, propagated, str(destination / "summary.png"), "\n".join(lines)
    except (ValueError, RuntimeError) as exc:
        raise gr.Error(str(exc)) from exc


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="FLUX.2 Klein — Scribble Guidance") as demo:
        gr.Markdown(
            "## FLUX.2 Klein — Draw-to-edit\n"
            "Upload an image, draw a red scribble, enter a prompt containing the object name, and click Generate."
        )
        with gr.Row():
            with gr.Column(scale=1):
                editor = gr.ImageEditor(
                    label="Input image + scribble (draw in red)", type="pil", image_mode="RGBA",
                    sources=["upload", "clipboard"],
                    brush=gr.Brush(default_size=12, colors=["#FF0000"], default_color="#FF0000", color_mode="fixed"),
                    layers=False, transforms=(), height=440,
                )
                with gr.Row():
                    check = gr.Button("Preview detected scribble", size="sm")
                    clear = gr.Button("Clear scribble (keep image)", size="sm")
                prompt = gr.Textbox(label="Editing prompt", placeholder="Insert a winding red snake following the drawn line", lines=2)
                object_phrase = gr.Textbox(label="Object phrase in the prompt", placeholder="snake")
                with gr.Accordion("Pipeline + guidance settings", open=False):
                    mode = gr.Dropdown(["base", "inpaint"], value="base", label="Pipeline mode", info="Base = normal Klein edit loop. Inpaint = use Klein inpaint pipeline and preserve outside the scribble mask.")
                    inpaint_strength = gr.Slider(minimum=0.1, maximum=1.0, value=0.8, step=0.05, label="Inpaint strength", info="Only for inpaint mode. Higher = start from noisier latents and allow larger edits.")
                    max_side = gr.Slider(minimum=128, maximum=1024, value=512, step=16, label="Maximum image side (px)", info="Image is resized so its long side does not exceed this value.")
                    steps = gr.Slider(minimum=4, maximum=60, value=30, step=1, label="Denoising steps", info="Total sampling steps. More steps are slower but may improve quality.")
                    guide_steps = gr.Slider(minimum=0, maximum=30, value=10, step=1, label="Early steps with guidance", info="How many early denoising steps receive scribble guidance.")
                    strength = gr.Slider(minimum=0.0, maximum=0.1, value=0.01, step=0.001, label="Latent guidance strength", info="Step size for the latent update from the guidance gradient.")
                    cfg = gr.Slider(minimum=1, maximum=8, value=4, step=0.5, label="CFG scale", info="Classifier-free guidance scale for prompt adherence.")
                    blocks = gr.Textbox(label="Attention blocks", value="D0,D1,D2,D3,D4", info="Blocks used to collect image-to-text attention for the guidance loss.")
                    seed = gr.Number(label="Random seed", value=42, precision=0, info="Use the same seed to compare settings.")
                    with gr.Row():
                        device = gr.Dropdown(["auto", "cuda", "mps", "cpu"], value="auto", label="Device", info="Where to run the model.")
                        dtype = gr.Dropdown(["auto", "bf16", "fp32"], value="auto", label="Model dtype", info="bf16 is faster/lighter on supported GPUs. fp32 is more stable but heavier.")
                    cpu_offload = gr.Checkbox(value=False, label="CPU offload (CUDA only)", info="Moves parts of the model between CPU and GPU to save VRAM.")
                with gr.Accordion("Attention head weighting", open=False):
                    gr.Markdown(
                        "**Uniform** reproduces the old mean across attention heads. "
                        "**Adaptive** favors heads whose scribble alignment improves after the "
                        "current latent update. Weights apply on the **next** guided step. "
                        "This measures responsiveness, not causal influence on object geometry."
                    )
                    attention_weighting = gr.Dropdown(
                        ["uniform", "adaptive"], value="uniform", label="Head weighting mode",
                        info="Uniform = old baseline. Adaptive = learn per-head weights from same-timestep before/after attention loss."
                    )
                    head_score_metric = gr.Dropdown(
                        ["selected", "coverage"], value="selected", label="Per-head improvement metric",
                        info="Selected = use your enabled loss combination. Coverage = use the old coverage loss only to evaluate heads, regardless of guidance losses."
                    )
                    head_temperature = gr.Number(
                        value=0.01, label="Softmax temperature (tau)",
                        info="Smaller values focus on heads with larger loss reductions; larger values give more equal weights. Must be > 0."
                    )
                    head_ema = gr.Slider(
                        minimum=0.05, maximum=1.0, step=0.05, value=0.5, label="Weight update rate (EMA)",
                        info="1 = use most recent improvement scores; smaller values keep more of the previous weights."
                    )
                    head_uniform_mix = gr.Slider(
                        minimum=0, maximum=1, step=0.05, value=0.1, label="Minimum uniform mixing",
                        info="Keeps some weight on every head to reduce early collapse. 0 = only improvement-based weights; 1 = always uniform."
                    )
                with gr.Accordion("Loss settings", open=False):
                    use_coverage = gr.Checkbox(value=False, label="Use old coverage loss", info="The original simple loss: make attention cover the scribble region.")
                    use_focal = gr.Checkbox(value=True, label="Use ScribbleDiff focal loss", info="Pixel-wise focal objective on attention vs. the scribble mask.")
                    use_moment = gr.Checkbox(value=True, label="Use ScribbleDiff moment loss", info="Matches centroid and orientation of the attention map to the scribble.")
                    use_centerline = gr.Checkbox(value=True, label="Use centerline loss", info="Penalizes attention far from the scribble centerline.")
                    with gr.Row():
                        coverage_weight = gr.Number(value=1.0, label="Coverage weight", info="Weight of the old coverage loss in the total loss.")
                        focal_alpha = gr.Number(value=0.25, label="Focal alpha", info="Class-balancing term in focal loss.")
                        focal_beta = gr.Number(value=2.0, label="Focal beta", info="Focusing term in focal loss.")
                        focal_weight = gr.Number(value=3.0, label="Focal weight", info="Weight of the focal loss in the total loss.")
                    with gr.Row():
                        moment_alpha = gr.Number(value=1.0, label="Moment angle alpha", info="Extra weight on the orientation part inside the moment loss.")
                        moment_weight = gr.Number(value=1.0, label="Moment weight", info="Weight of the moment loss in the total loss.")
                        centerline_weight = gr.Number(value=1.0, label="Centerline weight", info="Weight of the centerline loss in the total loss.")
                with gr.Accordion("Self-attention scribble propagation", open=True):
                    propagate = gr.Checkbox(value=True, label="Enable scribble propagation", info="Grow sparse scribbles into a larger target region using self-attention.")
                    with gr.Row():
                        prop_start = gr.Slider(minimum=0, maximum=30, value=3, step=1, label="Start step", info="First timestep where propagation is allowed.")
                        prop_end = gr.Slider(minimum=0, maximum=30, value=10, step=1, label="End step (exclusive)", info="Propagation stops before this timestep.")
                    prop_blocks = gr.Textbox(label="Self-attention blocks", value="D2,D3", info="Blocks used to extract self-attention for propagation.")
                    with gr.Row():
                        prop_top_k = gr.Slider(minimum=1, maximum=64, value=12, step=1, label="Max new cells per step", info="Limits how many new grid cells can be added per propagation step.")
                        prop_threshold = gr.Slider(minimum=0.001, maximum=2.0, value=0.2, step=0.001, label="Maximum symmetric KL", info="Smaller values make propagation stricter.")
                    prop_max_area = gr.Slider(minimum=0.05, maximum=1.0, value=0.3, step=0.05, label="Maximum expanded grid fraction", info="Upper bound on total propagated area as a fraction of the attention grid.")
                submit = gr.Button("Generate edited image", variant="primary")
            with gr.Column(scale=1):
                output = gr.Image(label="Generated image", type="pil")
                mask_preview = gr.Image(label="Original scribble mask", type="pil", image_mode="L")
                prop_preview = gr.Image(label="Propagated scribble mask", type="pil", image_mode="L")
                summary = gr.Image(label="Attention and result summary")
                status = gr.Textbox(label="Run details", interactive=False, lines=5)

        check.click(preview_stroke, inputs=[editor], outputs=[mask_preview])

        def reset_strokes(value):
            if not value or value.get("background") is None:
                return None
            bg = value["background"]
            return {"background": bg, "layers": [], "composite": bg}

        clear.click(reset_strokes, inputs=[editor], outputs=[editor])
        submit.click(
            generate,
            inputs=[
                editor, prompt, object_phrase, max_side, steps, guide_steps, strength,
                cfg, blocks, seed, device, dtype, cpu_offload,
                mode, inpaint_strength,
                attention_weighting, head_score_metric, head_temperature, head_ema, head_uniform_mix,
                use_coverage, use_focal, use_moment, use_centerline,
                coverage_weight, focal_alpha, focal_beta, focal_weight,
                moment_alpha, moment_weight, centerline_weight,
                propagate, prop_start, prop_end, prop_blocks, prop_top_k,
                prop_threshold, prop_max_area,
            ],
            outputs=[output, mask_preview, prop_preview, summary, status],
            concurrency_limit=1,
        )
    return demo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--inbrowser", action="store_true")
    args = parser.parse_args()
    build_ui().queue().launch(server_name=args.host, server_port=args.port, inbrowser=args.inbrowser, share=True)


if __name__ == "__main__":
    main()
