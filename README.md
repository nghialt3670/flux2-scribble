# FLUX.2 Klein Base — scribble attention guidance

A **research baseline** for user-drawn scribble guidance during image editing, now with optional self-attention scribble propagation.

- **Reference**: `black-forest-labs/FLUX.2-klein-base-4B` (the BASE variant, not the 4-step distilled model)
- **Libraries**: PyTorch + the current Hugging Face Diffusers Git source
- **Guidance**: Uniform average of object-token image-to-text attention from selected transformer blocks, simple differentiable scribble coverage loss, RMS-normalized gradient descent on the current noisy latent, then native Euler step
- **Propagation**: optional region growth via output-image self-attention similarity (4-connected frontier, symmetric KL, top-k). Expansion updates the coverage-loss target during early denoising.
- **No training**; no per-head adaptive weights yet
- **No inpainting mask**: output is reference-conditioned image editing, not strict background preservation

## Setup

Python 3.11–3.13 and `uv` are required. Model weights download on first use.

```bash
uv sync
```

If CUDA is available, run on a high-memory NVIDIA GPU. On an Apple M-series Mac the code can use `mps`, but gradient-based 4B inference can be memory-heavy and significantly slower than ordinary inference. A 48GB unified-memory Mac may still run out of memory.

## Draw your scribble in a browser (recommended)

```bash
uv sync
uv run python ui.py --inbrowser
```

Open `http://127.0.0.1:7861` (if it does not open automatically).

1. Upload or paste the original image into the canvas.
2. Draw a **red** stroke over the image with the brush (erase/undo if needed).
3. Enter a prompt such as `Insert a curved cat following the drawn line`, with `cat` as the object phrase.
4. Under **Self-attention scribble propagation**, keep *Enable scribble propagation* checked (default). Choose the step window and the KL threshold, then click **Generate edited image**.
5. Inspect the **Propagated scribble mask** output and `propagation_history/` files to verify how the mask grew.

The UI sends the *clean original background* as the reference image, and derives a separate grayscale scribble mask from the transparency of the drawing layer. The red annotation is **not** baked into the model's reference image. The model loads on the first generation. Results and diagnostics are saved under `outputs/ui_...`. The UI and CLI share the same latent-guidance pipeline.

## Inputs (CLI alternative)

- `input.png`: input/reference image
- `scribble.png`: **same pixel dimensions**. White strokes on black background by default. For black-on-white strokes use `--polarity dark`. RGBA transparent overlay: `--polarity alpha`.
- `--object`: **exact phrase appearing in the prompt**, e.g. `cat` in `Insert a cat following the drawn curve`.

## Run

```bash
uv run python app.py \
  --image input.png \
  --scribble scribble.png \
  --prompt "Insert a curved cat following the user's drawn stroke" \
  --object "cat" \
  --max-side 512 --steps 30 --guide-steps 10 \
  --strength 0.01 --blocks D0,D1,D2,D3,D4 \
  --propagate --prop-start 3 --prop-end 10 \
  --prop-blocks D2,D3 --prop-top-k 12 \
  --prop-threshold 0.2 --prop-max-area 0.30 \
  --out outputs
```

To check ordinary Klein inference with the same edit setup (guidance disabled):

```bash
uv run python app.py --image input.png --scribble scribble.png \
  --prompt "Insert a cat" --object "cat" --guide-steps 0 \
  --steps 30 --out outputs_baseline
```

For debugging, try a single block and a smaller native image:

```bash
uv run python app.py --image input.png --scribble scribble.png \
  --prompt "Insert a cat" --object "cat" \
  --max-side 256 --blocks D2 --steps 12 --guide-steps 3 \
  --out outputs_debug
```

## Propagation algorithm (ScribbleDiff-inspired Klein adaptation)

The U-Net paper's propagation uses self-attention distributions to grow the scribble into similar nearby spatial regions. FLUX.2 Klein does **not** have the U-Net's separate self-attention layers; its transformer has joint attention. We therefore construct a spatial self-attention distribution by using **only output-image queries and output-image keys** in selected transformer blocks, normalize each query row, and average heads and selected blocks. Text and reference-image keys are excluded from the region-growing comparison. The native model forward remains unmodified.

On steps `--prop-start <= i < --prop-end` (within guided steps):

1. Collect image-to-image attention from the regular guidance forward pass, with no separate model forward required.
2. Average attention rows from the **original scribble pixels** into a reference distribution.
3. Test 4-connected unvisited neighbors of the currently expanded scribble. A candidate's attention distribution must have symmetric KL divergence below `--prop-threshold`.
4. Add at most `--prop-top-k` new pixels on the image-token grid, with an overall area cap `--prop-max-area`.
5. Update the scribble target of the coverage loss, and continue normal gradient-guided denoising.

The default `--prop-threshold 0.2` is **not the paper's 0.0001 setting**: the paper measures divergence on transformed U-Net attention maps, whereas our Klein adaptation measures it directly on normalized output-image spatial distributions. This threshold is experimental and may need tuning. Unlike the original code, we explicitly anchor the prototype to the original stroke to reduce drift and cap its growth to help protect background regions.

To open the Gradio UI from a RunPod instance, use for example:

```bash
uv run python ui.py --host 0.0.0.0 --port 7862
```

Expose TCP/HTTP port 7862 in your RunPod configuration and use its proxy URL; `0.0.0.0` is the server binding address, not an address to open in your browser.

## Outputs

| File | Purpose |
| --- | --- |
| `result.png` | Final edit resized to match original input dimensions |
| `result_native.png` | Final edit at actual inference resolution (multiples of 16) |
| `summary.png` | Input / scribble / final attention / edit |
| `attention_last_guided.npy` | Last guided map, image-token resolution |
| `guidance_log.json` | Per-step loss and gradient RMS |
| `input_processed.png` / `scribble_processed.png` | Inputs at actual inference resolution |
| `scribble_propagated.png` | Final expanded scribble (when enabled) |
| `propagation_history/step_XXX.png` | Expanded mask at each propagation timestep |
| `guidance_log.json` `propagation` field | Number of new cells, current area, smallest symmetric KL |

## Structure

- `app.py`: CLI, model loading, same-size input preprocessing, saves output
- `ui.py`: Gradio draw-on-image interface; returns original background plus stroke-alpha mask
- `input_utils.py`: shared resizing and input validation
- `pipeline.py`: reference-conditioning, guidance, CFG, native Euler sampling and VAE decoding
- `attention.py`: **differentiable** attention extraction using Q/K and RoPE forward pre-hooks (D and S blocks supported)
- `guidance.py`: stroke distribution, cross-entropy loss, gradient normalization
- `visualize.py`: attention diagnostics
- `propagation.py`: image-to-image self-attention propagation with 4-connected growing

## Important implementation choices

1. Only **output-image queries** (`text_len : text_len + output_latent_tokens`) are used, never reference-image queries.
2. Text phrase positions are found using Qwen tokenizer offsets in the **actual chat template**; not guessed token indices.
3. Each map is normalized using all keys in the native attention (text + output + reference image). Only the target object text key positions are summed.
4. Attention processors are not replaced: Q/K attention maps are recomputed in pre-hooks, while the model retains its own native attention forward.
5. The model weights are frozen. Gradient is taken only with respect to the noisy image latent.
6. Use the original base scheduler sigma schedule and **recompute velocity after each guided latent update**.
7. Guidance is **not** ScribbleDiff's exact focal+moment loss. This intentionally implements a simple object-token coverage baseline first.
8. Propagation is adapted from ScribbleDiff to joint attention and is therefore **not a bit-for-bit reproduction**. Current weights are equal across selected blocks and heads. Single-stream blocks can be tested with `--blocks S0,S4,S9`, at higher memory cost.

## Status and caveats

Source code is syntax-checked; propagation and Q/K attention-capture helpers have CPU unit tests, and the Gradio UI builds successfully. An end-to-end FLUX.2 checkpoint execution was **not available** in the creation environment. Treat this as a research implementation to run and validate on your machine. The latest Diffusers internals may change; if model API errors occur, pin the working Diffusers revision in `pyproject.toml`.

For high VRAM use, start at `--max-side 256 --blocks D2 --guide-steps 2`. With GPU, gradient guidance is still noticeably costlier than plain inference, especially for single-stream layers.

References:
- https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux2/pipeline_flux2_klein.py
- https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_flux2.py
- https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B

- https://github.com/kaist-cvml/scribble-guided-diffusion/blob/main/scribble_propagation.py
