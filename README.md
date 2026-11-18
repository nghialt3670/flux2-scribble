# FLUX.2 Klein 4B Base — scribble-guided image editing

Custom research guidance loop for image editing with four selectable losses, optional scribble propagation, and **adaptive attention-head weighting**. Both CLI and Gradio UI are included. No model training required.

## Setup and browser UI

```bash
uv sync
uv run python ui.py --host 0.0.0.0 --port 7861
```

Upload the source image, draw the red scribble directly on the image, enter a prompt and its target object phrase, then Generate. The original image is passed to Klein **without the red strokes**. On a remote instance, expose port `7861` and open your provider's proxy URL.

## What the attention weighting does

The original baseline **uniform** averages object-token attention over every head of each selected transformer block.

The new **adaptive** option keeps each attention head separately. At guided timestep `t`:

1. Construct the current weighted image-to-object-text attention map: `A = sum_h w[h] * A[h]`.
2. Optimize the noisy image latent using the enabled losses on `A` (the same gradient update as before).
3. At the **same timestep**, collect individual attention maps at the *updated* latent. This reuses Klein's ordinary conditional prediction forward pass, which is already required before the Euler step.
4. Score each head **before and after** the latent update: `improvement[h] = L_h(before) - L_h(after)`.
5. For the next guided step, compute `proposal = softmax(improvement / temperature)`, mix a fraction of uniform weights, then blend with the previous weights using EMA.

The first guided step always starts uniformly. Weights are applied to the **next** guided step, not retroactively to the current step. Weighting spans all selected heads across all selected blocks, e.g. `D2:H03`.

**Scoring modes:**

- **selected** (default): head responsiveness to the exact enabled loss combination (coverage, focal, moment, centerline, including their weights).
- **coverage**: head responsiveness measured using the old coverage loss only, regardless of which losses are enabled for the actual latent update. This is useful for experiments based on the earlier coverage baseline.

**Parameters:**

- **Softmax temperature** (`0.01` default): lower values emphasize heads with larger same-step improvements. Adjust based on the measured improvement scale.
- **EMA update rate** (`0.5`): higher values react more quickly; lower values smooth noisy measurements.
- **Uniform mix** (`0.1`): keeps some attention weight on less-responsive heads to avoid premature collapse. Set to `0` for pure adaptive softmax.

**Important:** Improvement after a shared latent perturbation is a measure of head *responsiveness*, **not causal attribution** of shape control. Capturing per-head Q/K maps adds computation and memory compared with uniform mode, even though no additional model forward/backward is required.

## Losses (independently switchable)

| Loss | Default | Purpose |
| --- | --- | --- |
| Old **coverage** | off | Spatial cross-entropy between normalized scribble and object attention distributions |
| ScribbleDiff-style **focal** | on | Penalize per-location mismatch, with emphasis determined by alpha/beta |
| ScribbleDiff-style **moment** | on | Match scribble/attention centroid and orientation |
| New **centerline** | on | Penalize object attention far from the scribble line |

You can use any nonempty weighted combination of these losses. `guidance_log.json` records loss components per step.

The ScribbleDiff moment objective is **adapted** to the object-token attention map because the original paper's separate object-region mask is unavailable. The centerline loss is an expected distance-to-scribble surrogate, not a direct generated-object skeleton comparison.

## CLI examples

**Uniform baseline:**

```bash
uv run python app.py \
  --image input.png --scribble scribble.png \
  --prompt "Insert a curved cat following the scribble" --object "cat" \
  --mode base --attention-weighting uniform --steps 30 --guide-steps 10 \
  --strength 0.01 --blocks D0,D1,D2,D3,D4 --out outputs_uniform
```

**Adaptive, using only the old loss for both guidance and head scoring:**

```bash
uv run python app.py \
  --image input.png --scribble scribble.png \
  --prompt "Insert a curved cat following the scribble" --object "cat" \
  --mode base --steps 30 --guide-steps 10 --strength 0.01 \
  --blocks D0,D1,D2,D3,D4 --attention-weighting adaptive \
  --head-score-metric coverage --head-temperature 0.01 \
  --head-ema 0.5 --head-uniform-mix 0.1 \
  --use-coverage --no-focal --no-moment --no-centerline \
  --out outputs_adaptive
```

**Adaptive, using the selected loss combination to evaluate heads:**

```bash
uv run python app.py \
  --image input.png --scribble scribble.png \
  --prompt "Insert a curved cat following the scribble" --object "cat" \
  --mode base --attention-weighting adaptive --head-score-metric selected \
  --steps 30 --guide-steps 10 --out outputs_adaptive_selected
```

To switch to the **experimental inpaint pipeline**, use `--mode inpaint --inpaint-strength 0.8`. This custom guided inpaint path remains unverified on the full checkpoint. The separate scribble mask doubles as the inpaint region; narrow strokes may restrict edits too much, and propagation can widen the region.

For black-on-white scribbles, use `--polarity dark`. The CLI accepts a mask matching the original image dimensions; the UI draws it directly.

## Outputs

| File | Description |
| --- | --- |
| `result.png` | Final result at original input dimensions |
| `result_native.png` | Native inference dimensions |
| `summary.png` | Input, scribble, attention, generated image |
| `guidance_log.json` | Timesteps, loss components, gradient RMS, top-weighted heads |
| `attention_last_guided.npy` | Final guided attention map (token grid) |
| `attention_weight_history.json` | **Adaptive only**: all head labels, weights used, next-step weights, improvements and effective head count for each guided step |
| `scribble_propagated.png` | Final propagation mask if enabled |
| `propagation_history/` | Masks from each propagation step |

## Structure and validation

- `attention.py`: per-head differentiable attention extraction (D and S blocks) and self-attention maps for propagation.
- `attention_weighting.py`: per-head alignment scoring, softmax / EMA updates, weighting.
- `guidance.py`: coverage, focal, moment, centerline losses and latent gradient update.
- `pipeline.py`: training-free denoising with base/inpaint modes and online attention weighting.
- `ui.py`: Gradio ImageEditor and all settings with explanations.
- `app.py`: CLI.
- `tests/`: CPU tests including rectangular maps, loss-score equivalence, head capture and gradient propagation.

Validated without model weights: Python syntax checks, Gradio UI construction, and CPU tests. **Full FLUX.2 Klein checkpoint inference and resulting image quality have not been validated in this environment.** Current Diffusers Git HEAD may introduce API changes; pin a known-good revision once your run works.

## Individual attention-head control (new)

In Gradio, expand **Attention head configuration**. Three mutually exclusive modes:

| Head mode | What enters the guidance loss |
| --- | --- |
| **Uniform** | The original equal-head / equal-block attention average (unchanged baseline). |
| **Adaptive weighting** | All heads in `Attention blocks` with online before/after loss-improvement weights. |
| **Individual heads** | **Only** the explicitly named heads, with fixed relative weights. Unlisted heads have zero weight. |

In **Individual heads**, write for example:

```text
D2:H3,D2:H7=2,S4:H1=0.5
```

This selects head 3 in double-stream block 2 (weight 1), head 7 in D2 (weight 2), and head 1 in single-stream block 4 (weight 0.5). The program normalizes these values to sum to 1. It discovers the needed attention blocks from this list, ignoring the regular `Attention blocks` field. Head indices are zero-based. It rejects duplicate or out-of-range heads.

**These weights apply to the attention maps used by the guidance loss; they do not modify the transformer weights or native attention operators.** The latent still receives the gradient of the selected, weighted attention map.

**CLI example (manual):**

```bash
uv run python app.py \
  --image input.png --scribble scribble.png \
  --prompt "Insert a curved cat following the scribble" --object "cat" \
  --attention-weighting manual \
  --manual-heads "D2:H3,D2:H7=2,S4:H1=0.5" \
  --out outputs_manual
```

### Head statistics and visualization (new)

The UI defaults to **Save per-head statistics and heatmaps**. Disable it for faster runs. The CLI equivalent is on by default; `--no-head-statistics` disables it. `--head-vis-interval 3` exports maps on steps 0, 3, 6, ... **and the last guided step**. Use `--head-vis-interval 1` for every guided step.

Under `outputs.../attention_heads/`:

| File / folder | Purpose |
| --- | --- |
| `head_steps.csv` | Every captured head at **every guided timestep**: weight used, alignment loss, raw attention statistics, spatial entropy, scribble mass, peak token coordinates. Adaptive mode also includes same-step after-guidance loss and improvement. |
| `head_summary.csv` | One row per head: mean weight, mean loss, mean stroke mass, mean entropy and (adaptive mode) improvement. |
| `head_maps/step_XXX/<BLOCK>_HYY.png` | Separate attention-map overlay for **every** head in the selected blocks at each exported timestep. |
| `contact_sheets/step_XXX_<BLOCK>.png` | Side-by-side images of every head in a block (also previewed in the UI). |
| `head_trends/<BLOCK>_HYY.png` | Individual head chart: alignment loss and guidance weight over all guided steps. |
| `raw_maps/step_XXX.npz` | Floating-point attention maps of every head at every guided step, plus names, weights and scribble grid for later analysis. |
| `README.txt` | Definition of exported metrics and important interpretation notes. |

For uniform mode the diagnostics include all heads in the selected blocks. For manual mode they also include **unused heads** from the named blocks, labeled `selected=0`, `weight_used=0`, to support comparisons. A map's colors are normalized **individually for display**, so brightness between images is not a direct comparison of attention magnitude; use CSV metrics for that.

Exporting each head is more expensive than a single averaged attention map. These are **image-to-target-text probability maps**, not ground-truth object segmentations. A stronger head score or higher weight is **not proof of causal control** of object geometry.

The previous propagation, loss toggles, and base/experimental inpaint modes are retained unchanged. Full checkpoint inference has not been tested in this creation environment.
