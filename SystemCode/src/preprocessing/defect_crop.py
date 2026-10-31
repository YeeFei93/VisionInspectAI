"""Defect-focused image crops derived from binary localization masks."""

from typing import Optional, Tuple

import numpy as np
from PIL import Image


def defect_crop_box(
    mask: np.ndarray,
    padding_ratio: float = 0.25,
    min_crop_fraction: float = 0.25,
) -> Optional[Tuple[int, int, int, int]]:
    """Return a padded square PIL crop box around non-zero mask pixels."""
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array")
    if padding_ratio < 0:
        raise ValueError("padding_ratio must be non-negative")
    if not 0 < min_crop_fraction <= 1:
        raise ValueError("min_crop_fraction must be in (0, 1]")

    rows, columns = np.nonzero(mask)
    if len(rows) == 0:
        return None

    height, width = mask.shape
    defect_width = int(columns.max() - columns.min() + 1)
    defect_height = int(rows.max() - rows.min() + 1)
    side = max(
        defect_width,
        defect_height,
        int(round(min(height, width) * min_crop_fraction)),
    )
    side = min(min(height, width), int(round(side * (1 + 2 * padding_ratio))))
    center_x = (float(columns.min()) + float(columns.max()) + 1) / 2
    center_y = (float(rows.min()) + float(rows.max()) + 1) / 2

    left = int(round(center_x - side / 2))
    top = int(round(center_y - side / 2))
    left = min(max(left, 0), max(0, width - side))
    top = min(max(top, 0), max(0, height - side))
    right = min(width, left + side)
    bottom = min(height, top + side)
    return left, top, right, bottom


def crop_to_defect(
    image: Image.Image,
    mask: np.ndarray,
    padding_ratio: float = 0.25,
    min_crop_fraction: float = 0.25,
) -> Image.Image:
    """Crop around a defect mask, or return the image unchanged if empty."""
    if mask.shape != (image.height, image.width):
        mask_image = Image.fromarray(mask.astype(np.uint8) * 255)
        mask = np.asarray(mask_image.resize(image.size, Image.NEAREST)) > 0
    box = defect_crop_box(mask, padding_ratio, min_crop_fraction)
    return image if box is None else image.crop(box)