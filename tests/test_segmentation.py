"""Unit tests for src/preprocessing/segmentation.py."""

import numpy as np
from PIL import Image

from src.preprocessing.segmentation import compute_foreground_mask


def _make_synthetic_screw_image(size: int = 64) -> Image.Image:
    """Light, uniform background with a small darker square 'object' in the
    center — the minimal case Otsu thresholding needs to separate."""
    arr = np.full((size, size), 220, dtype=np.uint8)
    quarter = size // 4
    arr[quarter : 3 * quarter, quarter : 3 * quarter] = 40
    return Image.fromarray(arr).convert("RGB")


def test_compute_foreground_mask_isolates_darker_object():
    image = _make_synthetic_screw_image(size=64)
    mask = compute_foreground_mask(image, size=64)

    assert mask.dtype == bool
    assert mask.shape == (64, 64)
    # The object (minority class) should be marked foreground.
    assert mask[32, 32]
    # A background corner should be marked background.
    assert not mask[2, 2]


def test_compute_foreground_mask_object_is_minority_of_pixels():
    image = _make_synthetic_screw_image(size=64)
    mask = compute_foreground_mask(image, size=64)
    assert mask.mean() < 0.5
