"""CLI: run FLUX.2 Klein Base editing with ScribbleDiff-style latent guidance."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from diffusers import Flux2KleinPipeline

from pipeline import generate_guided
from visualize import save_summary

MODEL = "black-forest-labs/FLUX.2-klein-base-4B"


def load_inputs(image_path: str, scribble_path: str, max_side: int, polarity: str):
    image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    scribble = ImageOps.exif_transpose(Image.open(scribble_path))
    original_size = image.size
    if scribble.size != image.size:
        raise ValueError(f"Scribble size {scribble.size} must match image size {image.size}")
    if polarity == "alpha":
        if scribble.mode != "RGBA":
            raise ValueError("--polarity alpha requires an RGBA scribble with a transparent background")
        mask = np.asarray(scribble.getchannel("A"), dtype=np.uint8)
    else:
        gray = np.asarray(scribble.convert("L"), dtype=np.uint8)
        mask = (gray >= 128 if polarity == "light" else gray <= 127).astype(np.uint8) * 255
    if not mask.any():
        raise ValueError("Empty scribble mask. For black strokes on white, use --polarity dark")
    # Keep aspect ratio approximately; round to the native 16-pixel multiple.
    scale = min(1.0, max_side / max(original_size))
    width = max(16, round(original_size[0] * scale / 16) * 16)
    height = max(16, round(original_size[1] * scale / 16) * 16)
    size = (width, height)
    image = image.resize(size, Image.Resampling.LANCZOS)
    mask_image = Image.fromarray(mask, "L").resize(size, Image.Resampling.NEAREST)
    mask_tensor = torch.from_numpy(np.asarray(mask_image).copy()).float()[None, None] / 255.0
    return image, mask_image, mask_tensor, original_size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Reference image path")
    parser.add_argument("--scribble", required=True, help="White-on-black scribble image (or RGBA alpha)")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--object", required=True, help="Exact substring in prompt to track, e.g. 'cat'")
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guide-steps", type=int, default=10)
    parser.add_argument("--strength", type=float, default=0.01)
    parser.add_argument("--cfg", type=float, default=4.0)
    parser.add_argument("--blocks", default="D0,D1,D2,D3,D4", help="Comma-separated D0..D4 and/or S0..S19")
    parser.add_argument("--polarity", choices=["light", "dark", "alpha"], default="light")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp32"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-text-tokens", type=int, default=256)
    parser.add_argument("--query-chunk", type=int, default=64)
    parser.add_argument("--cpu-offload", action="store_true", help="Enable model CPU offload (mostly CUDA)")
    args = parser.parse_args()

    if args.max_side < 32:
        parser.error("--max-side must be >= 32")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = (torch.float32 if device == "cpu" else torch.bfloat16) if args.dtype == "auto" else (
        torch.bfloat16 if args.dtype == "bf16" else torch.float32
    )
    image, mask, mask_tensor, original_size = load_inputs(
        args.image, args.scribble, args.max_side, args.polarity
    )
    blocks = [part.strip().upper() for part in args.blocks.split(",") if part.strip()]
    if len(blocks) != len(set(blocks)):
        parser.error("--blocks must not contain duplicates")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    image.save(out / "input_processed.png")
    mask.save(out / "scribble_processed.png")
    print(f"Loading {MODEL} on {device} ({dtype}) | native size {image.size}", flush=True)
    pipe = Flux2KleinPipeline.from_pretrained(MODEL, torch_dtype=dtype)
    if args.cpu_offload:
        pipe.enable_model_cpu_offload(device=device)
    else:
        pipe.to(device)
    pipe.transformer.eval()
    result, details = generate_guided(
        pipe, image, mask_tensor, args.prompt, args.object,
        num_steps=args.steps, guide_steps=args.guide_steps,
        guidance_strength=args.strength, cfg_scale=args.cfg,
        blocks=blocks, seed=args.seed, max_text_tokens=args.max_text_tokens,
        attention_query_chunk=args.query_chunk, debug_dir=out,
    )
    result.save(out / "result_native.png")
    # Output file matches input dimensions. If not divisible by 16, this final
    # resize is only for output dimensions; sampling was at native size above.
    if result.size != original_size:
        result.resize(original_size, Image.Resampling.LANCZOS).save(out / "result.png")
    else:
        result.save(out / "result.png")
    save_summary(image, mask, result, details["last_attention"], out / "summary.png")
    print(f"Saved outputs to {out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
