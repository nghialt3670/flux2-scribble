"""Image and scribble input validation shared by the CLI and Gradio UI."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image, ImageOps


def prepare_inputs(image: Image.Image, mask: Image.Image, max_side: int):
    """Resize the image and a nonempty 8-bit scribble together.

    Returns (processed_RGB, processed_L_mask, mask_tensor_BCHW, original_size).
    The output image can be resized back to original_size after inference.
    """
    if max_side < 32:
        raise ValueError("max_side must be >= 32")
    image = ImageOps.exif_transpose(image).convert("RGB")
    # An L mask represents stroke alpha/coverage; do not threshold it.
    mask = mask.convert("L")
    original_size = image.size
    if mask.size != original_size:
        raise ValueError(f"Scribble size {mask.size} must match image size {original_size}")
    if not np.asarray(mask).any():
        raise ValueError("Please draw a scribble on the image before generating")

    scale = min(1.0, max_side / max(original_size))
    width = max(16, round(original_size[0] * scale / 16) * 16)
    height = max(16, round(original_size[1] * scale / 16) * 16)
    native_size = (width, height)
    processed_image = image.resize(native_size, Image.Resampling.LANCZOS)
    processed_mask = mask.resize(native_size, Image.Resampling.NEAREST)
    mask_tensor = torch.from_numpy(np.asarray(processed_mask).copy()).float()[None, None] / 255.0
    return processed_image, processed_mask, mask_tensor, original_size
