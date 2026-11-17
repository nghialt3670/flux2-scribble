"""Gradio drawing UI for FLUX.2 Klein Scribble Guidance.

Run: uv run python ui.py
Upload a reference image, paint a red line directly on it, then click Generate.
Model weights are loaded on the first run, not when opening the UI.
"""
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
    """Extract original background and stroke alpha mask, never the composite.

    Gradio ImageEditor returns {background, layers, composite}. Every layer is
    a transparent RGBA painting surface; the alpha channels carry the strokes.
    Using the composite as the input would mistakenly bake red strokes into
    the source image, so we deliberately use *background* as the reference.
    """
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
        # A drawing layer is RGBA with transparent pixels around the strokes.
        if "A" not in layer.getbands():
            raise ValueError("Drawing layer has no transparency; use the ImageEditor brush")
        alpha = np.asarray(layer.getchannel("A"), dtype=np.uint8)
        stroke = np.maximum(stroke, alpha)
    if not stroke.any():
        raise ValueError("Draw a red scribble on the image (using the brush) before generating")
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


@functools.lru_cache(maxsize=1)
def _load_model(device_selection: str, dtype_name: str, cpu_offload: bool):
    # Import at use time so the drawing UI can start without loading 4B weights.
    from diffusers import Flux2KleinPipeline

    device, dtype = _device_and_dtype(device_selection, dtype_name)
    print(f"Loading {MODEL} on {device}, dtype={dtype}", flush=True)
    pipe = Flux2KleinPipeline.from_pretrained(MODEL, torch_dtype=dtype)
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
    propagate, prop_start, prop_end, prop_blocks_string, prop_top_k,
    prop_threshold, prop_max_area,
    progress=gr.Progress(),
):
    try:
        if not prompt or not prompt.strip():
            raise ValueError("Enter an editing prompt")
        if not object_phrase or object_phrase.lower() not in prompt.lower():
            raise ValueError("Object phrase must appear in the editing prompt (e.g. 'cat')")
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
        progress(0.1, desc="Loading FLUX.2 Klein (first run may download weights)")
        pipe = _load_model(device, dtype, bool(cpu_offload))
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
        )
        result.save(destination / "result_native.png")
        if result.size != original_size:
            result = result.resize(original_size, Image.Resampling.LANCZOS)
        result.save(destination / "result.png")
        save_summary(image, mask, result, info["last_attention"], destination / "summary.png")
        progress(1.0, desc="Complete")
        lines = [f"Saved to {destination}", f"Native inference resolution: {image.width}x{image.height}"]
        guided = [e for e in info["log"] if e["loss"] is not None]
        if guided:
            lines.append(f"Guided steps: {len(guided)}; initial loss: {guided[0]['loss']:.5f}; last logged loss: {guided[-1]['loss']:.5f}")
        else:
            lines.append("Guidance disabled")
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
        gr.Markdown("## FLUX.2 Klein — Draw-to-insert\nUpload an image, **draw a red scribble over it**, enter a prompt containing the object name, and click Generate. The red line is used for guidance, **not** fed into Klein as image pixels.")
        with gr.Row():
            with gr.Column(scale=1):
                editor = gr.ImageEditor(
                    label="Input image + scribble (draw in red)", type="pil", image_mode="RGBA",
                    sources=["upload", "clipboard"], brush=gr.Brush(default_size=12, colors=["#FF0000"], default_color="#FF0000", color_mode="fixed"),
                    layers=False, transforms=(), height=440,
                )
                with gr.Row():
                    check = gr.Button("Preview detected scribble", size="sm")
                    clear = gr.Button("Clear scribble (keep image)", size="sm")
                prompt = gr.Textbox(label="Editing prompt", placeholder="Insert a winding red snake following the drawn line", lines=2)
                object_phrase = gr.Textbox(label="Object phrase in the prompt", placeholder="snake")
                with gr.Accordion("Guidance settings", open=False):
                    max_side = gr.Slider(minimum=128, maximum=1024, value=512, step=16, label="Maximum image side (px)")
                    steps = gr.Slider(minimum=4, maximum=60, value=30, step=1, label="Denoising steps")
                    guide_steps = gr.Slider(minimum=0, maximum=30, value=10, step=1, label="Early steps with guidance")
                    strength = gr.Slider(minimum=0.0, maximum=0.1, value=0.01, step=0.001, label="Latent guidance strength")
                    cfg = gr.Slider(minimum=1, maximum=8, value=4, step=0.5, label="CFG scale")
                    blocks = gr.Textbox(label="Attention blocks", value="D0,D1,D2,D3,D4")
                    seed = gr.Number(label="Random seed", value=42, precision=0)
                    with gr.Row():
                        device = gr.Dropdown(["auto", "cuda", "mps", "cpu"], value="auto", label="Device")
                        dtype = gr.Dropdown(["auto", "bf16", "fp32"], value="auto", label="Model dtype")
                    cpu_offload = gr.Checkbox(value=False, label="CPU offload (CUDA only)")
                with gr.Accordion("Self-attention scribble propagation", open=True):
                    propagate = gr.Checkbox(value=True, label="Enable scribble propagation")
                    gr.Markdown("Grow the drawn stroke into nearby regions with similar **image-to-image self-attention**. End step is exclusive; expansion is limited by top-k and KL threshold.")
                    with gr.Row():
                        prop_start = gr.Slider(minimum=0, maximum=30, value=3, step=1, label="Start step")
                        prop_end = gr.Slider(minimum=0, maximum=30, value=10, step=1, label="End step (exclusive)")
                    prop_blocks = gr.Textbox(label="Self-attention blocks", value="D2,D3")
                    with gr.Row():
                        prop_top_k = gr.Slider(minimum=1, maximum=64, value=12, step=1, label="Max new cells per step")
                        prop_threshold = gr.Slider(minimum=0.001, maximum=2.0, value=0.2, step=0.001, label="Maximum symmetric KL")
                    prop_max_area = gr.Slider(minimum=0.05, maximum=1.0, value=0.3, step=0.05, label="Maximum expanded grid fraction")
                submit = gr.Button("Generate edited image", variant="primary")
            with gr.Column(scale=1):
                output = gr.Image(label="Generated image", type="pil")
                mask_preview = gr.Image(label="Original scribble mask", type="pil", image_mode="L")
                prop_preview = gr.Image(label="Propagated scribble mask", type="pil", image_mode="L")
                summary = gr.Image(label="Attention and result summary")
                status = gr.Textbox(label="Run details", interactive=False, lines=3)

        check.click(preview_stroke, inputs=[editor], outputs=[mask_preview])

        def reset_strokes(value):
            if not value or value.get("background") is None:
                return None
            bg = value["background"]
            return {"background": bg, "layers": [], "composite": bg}

        clear.click(reset_strokes, inputs=[editor], outputs=[editor])
        submit.click(
            generate,
            inputs=[editor, prompt, object_phrase, max_side, steps, guide_steps, strength,
                    cfg, blocks, seed, device, dtype, cpu_offload,
                    propagate, prop_start, prop_end, prop_blocks, prop_top_k,
                    prop_threshold, prop_max_area],
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
    build_ui().queue().launch(server_name=args.host, server_port=args.port, inbrowser=args.inbrowser)


if __name__ == "__main__":
    main()
