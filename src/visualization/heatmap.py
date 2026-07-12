"""Anomaly heatmap visualization: overlay a per-pixel anomaly score map on
top of the original image."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def normalize_map(
    anomaly_map: torch.Tensor, vmin: Optional[float] = None, vmax: Optional[float] = None
) -> np.ndarray:
    arr = anomaly_map.detach().cpu().numpy()
    vmin = float(arr.min()) if vmin is None else vmin
    vmax = float(arr.max()) if vmax is None else vmax
    if vmax - vmin < 1e-8:
        return np.zeros_like(arr)
    return np.clip((arr - vmin) / (vmax - vmin), 0, 1)


def normalize_map_threshold(
    anomaly_map: torch.Tensor,
    threshold: float,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cool_span: float = 0.4,
    warm_span: float = 0.4,
) -> np.ndarray:
    """Normalize with a visual "cliff" at the good/defective decision
    `threshold`: everything below it is compressed into the cool half of
    the colormap (0.0-`cool_span`), everything at/above it into the hot
    half (1-`warm_span`-1.0). A plain min-max stretch makes ordinary
    sub-threshold texture look alarmingly "warm"; anchoring to the
    threshold keeps the heatmap's colors visually aligned with the actual
    decision boundary — only genuinely anomalous (above-threshold) regions
    read as hot.
    """
    arr = anomaly_map.detach().cpu().numpy() if isinstance(anomaly_map, torch.Tensor) else np.asarray(anomaly_map)
    vmin = float(arr.min()) if vmin is None else vmin
    vmax = float(arr.max()) if vmax is None else vmax

    below = arr < threshold
    above = ~below

    low_range = max(threshold - vmin, 1e-8)
    high_range = max(vmax - threshold, 1e-8)

    normalized = np.zeros_like(arr, dtype=np.float64)
    normalized[below] = cool_span * np.clip((arr[below] - vmin) / low_range, 0, 1)
    normalized[above] = (1 - warm_span) + warm_span * np.clip((arr[above] - threshold) / high_range, 0, 1)

    return normalized


def make_overlay(
    original_image: Image.Image,
    anomaly_map: torch.Tensor,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    alpha: float = 0.45,
    threshold: Optional[float] = None,
):
    """Build the normalized anomaly map, a jet-colormap heatmap image, and an
    original+heatmap overlay (all as uint8 numpy arrays). Shared by the batch
    figure-saving helper below and the interactive Streamlit demo.

    If `threshold` is given, colors are anchored to the good/defective
    decision boundary (see `normalize_map_threshold`) instead of a plain
    min-max stretch.
    """
    if threshold is not None:
        normalized = normalize_map_threshold(anomaly_map, threshold, vmin=vmin, vmax=vmax)
    else:
        normalized = normalize_map(anomaly_map, vmin=vmin, vmax=vmax)
    image_np = np.array(original_image.convert("RGB"))

    cmap = plt.get_cmap("jet")
    heatmap_rgb = (cmap(normalized)[..., :3] * 255).astype(np.uint8)
    overlay = (alpha * heatmap_rgb + (1 - alpha) * image_np).astype(np.uint8)

    return normalized, heatmap_rgb, overlay


def save_anomaly_heatmap(
    original_image: Image.Image,
    anomaly_map: torch.Tensor,
    output_path: Path,
    score: Optional[float] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    alpha: float = 0.45,
    gt_mask: Optional[np.ndarray] = None,
    threshold: Optional[float] = None,
) -> None:
    """Save a side-by-side figure: original | predicted heatmap | (ground-
    truth mask, if provided) | overlay. Passing `gt_mask` lets you visually
    compare the predicted anomaly heatmap against the true defect region.
    Passing `threshold` anchors the heatmap colors to the decision boundary
    (see `normalize_map_threshold`)."""
    normalized, _heatmap_rgb, overlay = make_overlay(
        original_image, anomaly_map, vmin=vmin, vmax=vmax, alpha=alpha, threshold=threshold
    )

    n_panels = 4 if gt_mask is not None else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4))

    axes[0].imshow(np.array(original_image.convert("RGB")))
    axes[0].set_title("Original")
    # `normalized` is already scaled into [0, 1] (and threshold-compressed if a
    # threshold was given) — pass vmin/vmax=0/1 explicitly so imshow doesn't
    # silently auto-rescale it back to the full colormap range.
    axes[1].imshow(normalized, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Predicted anomaly heatmap")

    if gt_mask is not None:
        axes[2].imshow(gt_mask, cmap="gray")
        axes[2].set_title("Ground-truth mask")
        overlay_ax = axes[3]
    else:
        overlay_ax = axes[2]

    overlay_title = "Overlay" if score is None else f"Overlay (score={score:.3f})"
    overlay_ax.imshow(overlay)
    overlay_ax.set_title(overlay_title)

    for ax in axes:
        ax.axis("off")
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
