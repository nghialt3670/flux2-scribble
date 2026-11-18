"""Per-head logging and visualizations for FLUX.2 Klein scribble guidance.

All selected blocks' heads are logged at every guided step. Maps are saved at
configurable intervals plus the final guided step. Saves per-head PNG overlays,
per-block contact sheets and CSV rows for every head at every guided step.

This is *image-query -> target-text-token attention probability mass*, not a
segmentation map, and weighting these maps does not edit transformer weights.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from attention_weighting import HeadScoreConfig, score_head_maps


class HeadDiagnostics:
    def __init__(self, output_dir: str | Path, source_image: Image.Image,
                 grid_h: int, grid_w: int, *,
                 interval: int = 3, score_config: HeadScoreConfig | None = None):
        if interval < 1:
            raise ValueError("Head visualization interval must be >= 1")
        self.root = Path(output_dir) / "attention_heads"
        self.root.mkdir(parents=True, exist_ok=True)
        self.image = source_image.convert("RGB")
        self.h, self.w = grid_h, grid_w
        self.interval = int(interval)
        self.score_config = score_config or HeadScoreConfig()
        self.rows: list[dict] = []
        self.by_step: dict[int, list[int]] = {}
        self.last_atlas: Path | None = None
        self.labels: list[str] | None = None
        self.rendered_steps: list[int] = []

    @staticmethod
    def _colorized(raw: np.ndarray, size: tuple[int, int]) -> Image.Image:
        """Per-head min/max contrast visualization (do NOT compare brightness across heads)."""
        from matplotlib import colormaps
        low = float(np.min(raw))
        high = float(np.max(raw))
        rel = (raw - low) / (high - low + 1e-12)
        rgb = np.uint8(colormaps["inferno"](rel)[..., :3] * 255)
        return Image.fromarray(rgb, "RGB").resize(size, Image.Resampling.BILINEAR)

    def _write_maps(self, step: int, labels: list[str], maps: np.ndarray,
                    scores: np.ndarray, weights: np.ndarray) -> None:
        per_head_dir = self.root / "head_maps" / f"step_{step:03d}"
        per_block_dir = self.root / "contact_sheets"
        per_head_dir.mkdir(parents=True, exist_ok=True)
        per_block_dir.mkdir(parents=True, exist_ok=True)
        grouped: dict[str, list[Image.Image]] = defaultdict(list)
        cols = 4
        tile_w, tile_h = 256, 240
        for j, label in enumerate(labels):
            raw = maps[j].reshape(self.h, self.w)
            # Background source image on the left, heatmap overlay on the right.
            backdrop = self.image.resize((256, 208), Image.Resampling.BILINEAR)
            heat = self._colorized(raw, backdrop.size)
            merged = Image.blend(backdrop, heat, alpha=0.58)
            tile = Image.new("RGB", (tile_w, tile_h), "#151719")
            tile.paste(merged, (0, 28))
            d = ImageDraw.Draw(tile)
            d.text((6, 5), f"{label}  weight={weights[j]:.4f}", fill="white")
            d.text((6, 223), f"loss={scores[j]:.5f}  raw_max={raw.max():.4g}", fill="white")
            tile.save(per_head_dir / f"{label.replace(':', '_')}.png")
            grouped[label.split(":")[0]].append(tile)
        for block, tiles in grouped.items():
            nrows = (len(tiles) + cols - 1) // cols
            atlas = Image.new("RGB", (tile_w * cols, tile_h * nrows + 32), "#151719")
            ImageDraw.Draw(atlas).text((10, 10), f"{block}  step {step} — individually contrast-scaled maps", fill="white")
            for j, tile in enumerate(tiles):
                atlas.paste(tile, ((j % cols) * tile_w, 32 + (j // cols) * tile_h))
            path = per_block_dir / f"step_{step:03d}_{block}.png"
            atlas.save(path)
            # The last exported block is displayed in the UI preview.
            self.last_atlas = path
        self.rendered_steps.append(step)

    @torch.no_grad()
    def record(self, *, step: int, timestep: float, labels: list[str],
               maps: torch.Tensor, target: torch.Tensor, weights: torch.Tensor,
               is_last_guided: bool = False,
               before_scores: torch.Tensor | None = None) -> None:
        """Record all captured heads (including unused heads in manual mode)."""
        if maps.ndim != 3 or maps.shape[0] != 1 or maps.shape[2] != self.h * self.w:
            raise ValueError("Expected batch-one head maps [1,H,h*w]")
        if len(labels) != maps.shape[1] or len(labels) != weights.numel():
            raise ValueError("Head names/weights do not match captured attention heads")
        if self.labels is None:
            self.labels = list(labels)
        elif self.labels != list(labels):
            raise RuntimeError("Head labels changed during this run")
        if before_scores is None:
            before_scores = score_head_maps(maps.detach(), target, self.h, self.w, self.score_config)
        vals = maps[0].detach().float()
        n = vals.shape[1]
        prob = vals.clamp_min(1e-12)
        prob = prob / prob.sum(-1, keepdim=True).clamp_min(1e-12)
        scribble = (target[0] > 0).to(prob.device, prob.dtype)
        stroke_mass = (prob * scribble[None, :]).sum(dim=1)
        entropy = -(prob * prob.log()).sum(-1)
        peak = vals.argmax(-1)
        w = weights.detach().float().cpu().numpy()
        vals_cpu = vals.cpu().numpy()
        scores = before_scores[0].detach().float().cpu().numpy()
        # Preserve numerical maps for research/re-plotting even at timesteps
        # whose PNG visualization is skipped.
        raw_dir = self.root / "raw_maps"
        raw_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            raw_dir / f"step_{step:03d}.npz",
            head_labels=np.asarray(labels),
            maps=vals_cpu.reshape(len(labels), self.h, self.w),
            weights=w,
            scribble_grid=(target[0].detach().float().cpu().numpy().reshape(self.h, self.w) > 0),
            timestep=np.asarray(float(timestep), dtype=np.float32),
        )
        self.by_step[step] = []
        for j, label in enumerate(labels):
            row = {
                "step": int(step), "timestep": float(timestep), "head": label,
                "block": label.split(":")[0], "head_index": int(label.split("H")[-1]),
                "selected": int(w[j] > 0), "weight_used": float(w[j]),
                "loss_before": float(scores[j]), "loss_after": "", "improvement": "",
                "raw_attention_mean": float(vals_cpu[j].mean()),
                "raw_attention_max": float(vals_cpu[j].max()),
                "raw_attention_std": float(vals_cpu[j].std()),
                "attention_entropy": float(entropy[j].item()),
                "stroke_probability_mass": float(stroke_mass[j].item()),
                "peak_grid_x": int(peak[j].item() % self.w),
                "peak_grid_y": int(peak[j].item() // self.w),
            }
            self.by_step[step].append(len(self.rows))
            self.rows.append(row)
        if step % self.interval == 0 or is_last_guided:
            self._write_maps(step, labels, vals_cpu, scores, w)

    @torch.no_grad()
    def record_after(self, step: int, after_scores: torch.Tensor) -> None:
        """Adaptive mode only: add score after guidance, same step and timestep."""
        indices = self.by_step.get(step)
        if indices is None or after_scores.shape != (1, len(indices)):
            raise ValueError("After-scores cannot be matched to the recorded head list")
        after = after_scores[0].detach().float().cpu().tolist()
        for index, value in zip(indices, after):
            row = self.rows[index]
            row["loss_after"] = float(value)
            row["improvement"] = float(row["loss_before"] - value)

    @staticmethod
    def _draw_trend(rows: list[dict], path: Path) -> None:
        """Lightweight per-head timeline: alignment loss and guidance weight."""
        width, height = 540, 270
        im = Image.new("RGB", (width, height), "#151719")
        draw = ImageDraw.Draw(im)
        title = f"{rows[0]['head']}  |  {len(rows)} guided steps"
        draw.text((14, 12), title, fill="#ffffff")
        draw.text((14, 32), "alignment loss (orange), head weight (cyan); separate y-scales", fill="#cccccc")
        left, right, top, bottom = 48, width - 20, 64, height - 44
        draw.rectangle((left, top, right, bottom), outline="#686868", width=1)
        x_steps = [int(r['step']) for r in rows]
        x_min, x_max = min(x_steps), max(x_steps)
        loss = [float(r['loss_before']) for r in rows]
        weight = [float(r['weight_used']) for r in rows]

        def line(values, color, y_label):
            y_min, y_max = min(values), max(values)
            diff = max(y_max - y_min, 1e-12)
            pts = [(
                int(left + (step - x_min) / max(x_max - x_min, 1) * (right - left)),
                int(bottom - (val - y_min) / diff * (bottom - top)),
            ) for step, val in zip(x_steps, values)]
            if len(pts) > 1:
                draw.line(pts, fill=color, width=3)
            for x, y in pts:
                draw.ellipse((x-3, y-3, x+3, y+3), fill=color)
            draw.text((12, y_label), f"{y_min:.4g}..{y_max:.4g}", fill=color)

        line(loss, "#ffb366", 66)
        line(weight, "#58dafa", 84)
        draw.text((left, height - 30), f"step {x_min}", fill="#cccccc")
        draw.text((right - 75, height - 30), f"step {x_max}", fill="#cccccc")
        path.parent.mkdir(parents=True, exist_ok=True)
        im.save(path)

    def finalize(self) -> None:
        if not self.rows:
            return
        columns = list(self.rows[0])
        with (self.root / "head_steps.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(self.rows)
        grouped = defaultdict(list)
        for row in self.rows:
            grouped[row["head"]].append(row)
        summary = []
        for label, rows in grouped.items():
            self._draw_trend(rows, self.root / "head_trends" / f"{label.replace(':', '_')}.png")
            summary.append({
                "head": label, "block": rows[0]["block"],
                "selected": int(any(x["selected"] for x in rows)),
                "mean_weight": float(np.mean([x["weight_used"] for x in rows])),
                "mean_loss_before": float(np.mean([x["loss_before"] for x in rows])),
                "mean_stroke_mass": float(np.mean([x["stroke_probability_mass"] for x in rows])),
                "mean_attention_entropy": float(np.mean([x["attention_entropy"] for x in rows])),
                "mean_improvement": (float(np.mean([x["improvement"] for x in rows if x["improvement"] != ""]))
                                     if any(x["improvement"] != "" for x in rows) else ""),
            })
        with (self.root / "head_summary.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
        (self.root / "README.txt").write_text(
            "All heads in the selected transformer blocks are included.\n"
            "head_steps.csv: one row per head per guided timestep; weights are those USED for that step.\n"
            "loss_before: per-head alignment loss at z_t, before latent guidance (lower is better).\n"
            "loss_after/improvement: filled only for adaptive mode from the same-timestep post-update pass.\n"
            "stroke_probability_mass: fraction of spatially normalized target-token attention on the scribble.\n"
            "raw_attention_*: direct image-query-to-target-text probability mass; not spatially normalized.\n"
            "Maps use their OWN min/max contrast scale for visibility: do not compare brightness between heads.\n"
            "contact_sheets/: all heads in a block together. head_maps/: per-head overlays, sampled at configured interval.\n"
            "head_trends/: per-head line charts of alignment loss and guidance weight across guided steps.\n"
            "raw_maps/: compressed NPZ with all heads' floating-point maps, weights and scribble grid at EVERY guided step.\n"
            "No gradients or additional inference forwards are used for diagnostics.\n"
        )
