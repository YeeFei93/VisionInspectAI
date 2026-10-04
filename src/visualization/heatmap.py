"""Anomaly heatmap visualization: overlay a per-pixel anomaly score map on
top of the original image."""

from pathlib import Path
from typing import List, Optional

import cv2
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
    """Normalize with a visual "cliff" at the supplied localization
    `threshold`: everything below it is compressed into the cool half of
    the colormap (0.0-`cool_span`), everything at/above it into the hot
    half (1-`warm_span`-1.0). A plain min-max stretch makes ordinary
    sub-threshold texture look alarmingly "warm"; anchoring to the
    threshold keeps the heatmap's colors visually aligned with the actual
    localization boundary — only genuinely anomalous (above-threshold) regions
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

    If `threshold` is given, colors are anchored to that localization
    boundary (see `normalize_map_threshold`) instead of a plain min-max
    stretch.
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


def create_defect_type_overlay(
    original_image: Image.Image,
    regions: List,
    image_size: int,
    alpha: float = 0.5,
) -> tuple:
    """Create a color-coded segmentation overlay for multiple defect types.
    
    Takes a list of DefectRegion objects and assigns a distinct color to each
    defect type, creating a categorical overlay that highlights which types
    of defects are present and where.
    
    Args:
        original_image: PIL Image (must match image_size)
        regions: List of DefectRegion objects from defect_regions.classify_defect_regions()
        image_size: Size of the segmentation mask (should match original_image)
        alpha: Transparency of overlay (0=original only, 1=fully opaque colors)
    
    Returns:
        Tuple of (colored_mask, overlay_image, type_to_color_dict)
        - colored_mask: (H, W, 3) uint8 RGB array with color-coded regions
        - overlay_image: (H, W, 3) uint8 RGB blend of original and colored mask
        - type_to_color_dict: dict mapping defect_type -> (R, G, B) color tuple
    """
    # Categorical palette: tab10 gives 10 maximally distinct hues (unlike
    # tab20c, which groups colors in same-hue shade blocks of 4 and looks
    # near-identical for a handful of types). Fall back to an evenly spaced
    # HSV sweep if a category ever has more than 10 defect types.
    unique_types = sorted(set(r.defect_type for r in regions))
    if len(unique_types) <= 10:
        colors_float = plt.get_cmap("tab10")(np.linspace(0, 1, 10))[:, :3]
    else:
        colors_float = plt.get_cmap("hsv")(np.linspace(0, 1, len(unique_types), endpoint=False))[:, :3]
    colors_uint8 = (colors_float * 255).astype(np.uint8)
    
    # Map unique defect types to colors
    type_to_color = {
        defect_type: tuple(colors_uint8[i % len(colors_uint8)])
        for i, defect_type in enumerate(unique_types)
    }
    
    # Create empty colored segmentation mask
    colored_mask = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    
    # Fill each region with its defect type's color
    for region in regions:
        x, y, w, h = region.bbox
        # Ensure coordinates are integers
        x, y, w, h = int(x), int(y), int(w), int(h)
        color = type_to_color[region.defect_type]
        # colored_mask is RGB (blended with RGB image_np, shown via st.image as
        # RGB) -- pass the color through as-is, no BGR swap, so it matches the legend.
        color_rgb = (int(color[0]), int(color[1]), int(color[2]))
        # Draw filled rectangle in the color for this defect type
        cv2.rectangle(colored_mask, (x, y), (x + w, y + h), color_rgb, thickness=-1)
    
    # Blend with original image
    image_np = np.array(original_image.convert("RGB"))
    overlay = (alpha * colored_mask.astype(np.float32) + 
               (1 - alpha) * image_np.astype(np.float32)).astype(np.uint8)
    
    return colored_mask, overlay, type_to_color


def save_defect_type_heatmap(
    original_image: Image.Image,
    regions: List,
    image_size: int,
    output_path: Path,
    alpha: float = 0.5,
) -> None:
    """Save a multi-panel figure showing defect-type segmentation.
    
    Displays: original | colored segmentation | overlay, with a legend showing
    defect types, colors, and confidences.
    
    Args:
        original_image: PIL Image
        regions: List of DefectRegion objects (sorted by confidence, highest first)
        image_size: Size of segmentation mask
        output_path: Path to save the figure
        alpha: Overlay transparency
    """
    colored_mask, overlay, type_to_color = create_defect_type_overlay(
        original_image, regions, image_size, alpha=alpha
    )
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(np.array(original_image.convert("RGB")))
    axes[0].set_title("Original")
    axes[0].axis("off")
    
    # Colored segmentation with legend
    axes[1].imshow(colored_mask)
    axes[1].set_title("Defect Type Segmentation")
    axes[1].axis("off")
    
    # Create legend
    legend_elements = []
    for region in regions:
        color_rgb = tuple(c / 255.0 for c in type_to_color[region.defect_type])
        label = f"{region.defect_type} (conf: {region.confidence:.2f})"
        from matplotlib.patches import Patch
        legend_elements.append(Patch(facecolor=color_rgb, label=label))
    
    axes[1].legend(handles=legend_elements, loc="upper right", fontsize=9)
    
    # Overlay
    axes[2].imshow(overlay)
    axes[2].set_title(f"Overlay (α={alpha})")
    axes[2].axis("off")
    
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


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
    Passing `threshold` anchors the heatmap colors to the localization boundary
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
