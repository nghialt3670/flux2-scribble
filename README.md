# FLUX.2 Klein — scribble guidance

Updated baseline with:

- **ScribbleDiff-style focal loss**
- **ScribbleDiff-style moment loss** (centroid + orientation)
- **Centerline loss** (distance-to-scribble-centerline surrogate)
- **Optional self-attention scribble propagation**
- **Base / inpaint mode switch**
- **CLI + Gradio UI**

## Important notes

1. This is still a **research implementation**.
2. We do **not** have a separate object mask at inference time, so the released ScribbleDiff moment formula is adapted to the aggregated object-token attention map.
3. The new **inpaint mode** is an **approximate latent inpainting mode** for the custom guidance loop:
   - it loads the **inpaint pipeline class**,
   - initializes latents from the source image,
   - and blends outside-mask latents back toward the source image after each step.
4. Scribble propagation can still be used in either mode.

## Main files

- `app.py` — CLI
- `ui.py` — Gradio UI
- `pipeline.py` — guidance loop, base/inpaint mode
- `guidance.py` — focal + moment + centerline losses
- `attention.py` — differentiable attention extraction
- `propagation.py` — self-attention-based scribble propagation

## CLI examples

### Base mode

```bash
uv run python app.py \
  --image input.png \
  --scribble scribble.png \
  --prompt "Insert a curved cat following the scribble" \
  --object "cat" \
  --mode base \
  --steps 30 --guide-steps 10 \
  --strength 0.01 \
  --blocks D0,D1,D2,D3,D4 \
  --propagate \
  --out outputs_base
```

### Inpaint mode

```bash
uv run python app.py \
  --image input.png \
  --scribble scribble.png \
  --prompt "Insert a curved cat following the scribble" \
  --object "cat" \
  --mode inpaint \
  --inpaint-strength 0.8 \
  --steps 30 --guide-steps 10 \
  --strength 0.01 \
  --blocks D0,D1,D2,D3,D4 \
  --propagate \
  --out outputs_inpaint
```

### Disable selected losses

```bash
uv run python app.py \
  --image input.png \
  --scribble scribble.png \
  --prompt "Insert a curved cat following the scribble" \
  --object "cat" \
  --no-centerline
```

## Default loss formula used now


total loss =

- `3.0 * focal_loss`
- `+ 1.0 * moment_loss`
- `+ 1.0 * centerline_loss`

You can change these weights from CLI/UI.

## UI

```bash
uv run python ui.py --inbrowser
```

The UI now includes:

- base / inpaint mode switch
- on/off toggles for focal / moment / centerline losses
- loss weights and parameters
- propagation settings

## Outputs

- `result.png`
- `result_native.png`
- `summary.png`
- `guidance_log.json`
- `attention_last_guided.npy`
- `scribble_propagated.png` (if propagation enabled)

`guidance_log.json` now stores loss components for each guided step.

## Rectangular-image moment fix (2026-11-17)

`_moment_from_map` now constructs coordinate grids in `[H,W]` order and normalizes x and y centroids by width and height respectively. Tested with rectangular 21x32/32x21 and square grids including autograd.
