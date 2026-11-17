# FLUX.2 Klein Base — scribble attention guidance

A **minimal research baseline** for user-drawn scribble guidance during image editing.

- **Reference**: `black-forest-labs/FLUX.2-klein-base-4B` (the BASE variant, not the 4-step distilled model)
- **Libraries**: PyTorch + the current Hugging Face Diffusers Git source
- **Guidance**: Uniform average of object-token image-to-text attention from selected transformer blocks, simple differentiable scribble coverage loss, RMS-normalized gradient descent on the current noisy latent, then native Euler step
- **No training**; no per-head adaptive weights yet
- **No inpainting mask**: output is reference-conditioned image editing, not strict background preservation

## Setup

Python 3.11–3.13 and `uv` are required. Model weights download on first use.

```bash
uv sync
```

If CUDA is available, run on a high-memory NVIDIA GPU. On an Apple M-series Mac the code can use `mps`, but gradient-based 4B inference can be memory-heavy and significantly slower than ordinary inference. A 48GB unified-memory Mac may still run out of memory.

## Inputs

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

## Outputs

| File | Purpose |
| --- | --- |
| `result.png` | Final edit resized to match original input dimensions |
| `result_native.png` | Final edit at actual inference resolution (multiples of 16) |
| `summary.png` | Input / scribble / final attention / edit |
| `attention_last_guided.npy` | Last guided map, image-token resolution |
| `guidance_log.json` | Per-step loss and gradient RMS |
| `input_processed.png` / `scribble_processed.png` | Inputs at actual inference resolution |

## Structure

- `app.py`: CLI, model loading, same-size input preprocessing, saves output
- `pipeline.py`: reference-conditioning, guidance, CFG, native Euler sampling and VAE decoding
- `attention.py`: **differentiable** attention extraction using Q/K and RoPE forward pre-hooks (D and S blocks supported)
- `guidance.py`: stroke distribution, cross-entropy loss, gradient normalization
- `visualize.py`: attention diagnostics

## Important implementation choices

1. Only **output-image queries** (`text_len : text_len + output_latent_tokens`) are used, never reference-image queries.
2. Text phrase positions are found using Qwen tokenizer offsets in the **actual chat template**; not guessed token indices.
3. Each map is normalized using all keys in the native attention (text + output + reference image). Only the target object text key positions are summed.
4. Attention processors are not replaced: Q/K attention maps are recomputed in pre-hooks, while the model retains its own native attention forward.
5. The model weights are frozen. Gradient is taken only with respect to the noisy image latent.
6. Use the original base scheduler sigma schedule and **recompute velocity after each guided latent update**.
7. Guidance is **not** ScribbleDiff's exact focal+moment loss. This intentionally implements a simple object-token coverage baseline first.
8. Current weights are equal across selected blocks and heads. Single-stream blocks can be tested with `--blocks S0,S4,S9`, at higher memory cost.

## Status and caveats

Source code is syntax-checked and the mathematical helper is unit-tested here, but an end-to-end FLUX.2 checkpoint execution was **not available** in the creation environment. Treat this as a research implementation to run and validate on your machine. The latest Diffusers internals may change; if model API errors occur, pin the working Diffusers revision in `pyproject.toml`.

For high VRAM use, start at `--max-side 256 --blocks D2 --guide-steps 2`. With GPU, gradient guidance is still noticeably costlier than plain inference, especially for single-stream layers.

References:
- https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux2/pipeline_flux2_klein.py
- https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_flux2.py
- https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B
