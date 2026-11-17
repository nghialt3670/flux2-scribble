"""Save a quick attention/scribble comparison figure."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def save_summary(input_image: Image.Image, scribble_mask: Image.Image, generated: Image.Image,
                 attention, destination: str | Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    axes[0].imshow(input_image)
    axes[0].set_title("Input")
    axes[1].imshow(scribble_mask, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Scribble")
    if attention is not None:
        heat = np.asarray(attention.float())
        # Display at native 2D token grid over the reference image; no misleading
        # claim that heat is pixel-resolution object segmentation.
        axes[2].imshow(input_image)
        axes[2].imshow(
            heat, extent=(0, input_image.width, input_image.height, 0),
            interpolation="bilinear", alpha=0.65, cmap="inferno"
        )
        axes[2].set_title("Last guided attention")
    else:
        axes[2].imshow(input_image)
        axes[2].set_title("No attention (guidance off)")
    axes[3].imshow(generated)
    axes[3].set_title("Generated output")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)
