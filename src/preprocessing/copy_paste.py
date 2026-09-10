"""Mask-guided copy-paste synthesis of defective images.

Cuts a real defect region out of a labeled defective image using its MVTec
ground-truth mask and blends it onto a clean `train/good` image at a random
position/scale/rotation. Unlike text-prompted generative synthesis, the
pasted pixels *are* a real instance of the source image's `defect_type`, so
the synthetic image's fine-grained label is correct by construction --
which matters for categories whose defect types are only subtly different
from each other (e.g. screw's `thread_side` vs `thread_top`).

Follows the CutPaste / DRAEM / NSA family of synthetic-anomaly augmentation.
"""

from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter

DEFAULT_SCALE_RANGE = (0.7, 1.3)
DEFAULT_ROTATION_DEGREES = 20.0
DEFAULT_FEATHER_RADIUS = 2.0
DEFAULT_JITTER_RATIO = 0.08


def mask_centroid(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    """(x, y) centre of a mask's set pixels, or None when it is empty."""
    rows, columns = np.nonzero(mask)
    if len(rows) == 0:
        return None
    return float(columns.mean()), float(rows.mean())


def extract_defect_patch(
    image: Image.Image, mask: np.ndarray
) -> Optional[Tuple[Image.Image, np.ndarray]]:
    """Crop `image` and `mask` to the mask's tight bounding box.

    Returns None when the mask has no set pixels.
    """
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array")
    if mask.shape != (image.height, image.width):
        resized = Image.fromarray(mask.astype(np.uint8) * 255).resize(image.size, Image.NEAREST)
        mask = np.asarray(resized) > 0

    rows, columns = np.nonzero(mask)
    if len(rows) == 0:
        return None

    left, right = int(columns.min()), int(columns.max()) + 1
    top, bottom = int(rows.min()), int(rows.max()) + 1
    return image.crop((left, top, right, bottom)), mask[top:bottom, left:right]


def paste_defect(
    background: Image.Image,
    patch: Image.Image,
    patch_mask: np.ndarray,
    rng: np.random.Generator,
    scale_range: Tuple[float, float] = DEFAULT_SCALE_RANGE,
    rotation_degrees: float = DEFAULT_ROTATION_DEGREES,
    feather_radius: float = DEFAULT_FEATHER_RADIUS,
    anchor: Optional[Tuple[float, float]] = None,
    jitter_ratio: float = DEFAULT_JITTER_RATIO,
) -> Tuple[Image.Image, np.ndarray]:
    """Blend a defect patch into `background`.

    With `anchor` (an (x, y) point, normally the defect's centre in its
    source image), the patch is placed there plus a small random jitter;
    without it, placement is uniformly random. Anchoring matters for
    discrete-object categories -- a `bent_lead` defect dropped onto the bare
    circuit board instead of onto the transistor's leads is not a plausible
    example of that class.

    Returns the composited image and the boolean mask of the pasted region,
    so the result can be used with the same defect-focused cropping path as
    a real ground-truth-masked image.
    """
    if scale_range[0] <= 0 or scale_range[1] < scale_range[0]:
        raise ValueError("scale_range must be positive and ordered (low, high)")

    scale = float(rng.uniform(*scale_range))
    width = max(1, int(round(patch.width * scale)))
    height = max(1, int(round(patch.height * scale)))
    patch = patch.resize((width, height), Image.BILINEAR)
    mask_image = Image.fromarray((patch_mask > 0).astype(np.uint8) * 255).resize(
        (width, height), Image.NEAREST
    )

    if rotation_degrees:
        angle = float(rng.uniform(-rotation_degrees, rotation_degrees))
        patch = patch.rotate(angle, resample=Image.BILINEAR, expand=True)
        mask_image = mask_image.rotate(angle, resample=Image.NEAREST, expand=True)

    # A rotated/upscaled patch can exceed the background; shrink it to fit.
    fit = min(background.width / patch.width, background.height / patch.height, 1.0)
    if fit < 1.0:
        width = max(1, int(patch.width * fit))
        height = max(1, int(patch.height * fit))
        patch = patch.resize((width, height), Image.BILINEAR)
        mask_image = mask_image.resize((width, height), Image.NEAREST)

    max_x = background.width - patch.width
    max_y = background.height - patch.height
    if anchor is None:
        x = int(rng.integers(0, max_x + 1))
        y = int(rng.integers(0, max_y + 1))
    else:
        jitter = jitter_ratio * min(background.width, background.height)
        x = int(round(anchor[0] - patch.width / 2 + rng.uniform(-jitter, jitter)))
        y = int(round(anchor[1] - patch.height / 2 + rng.uniform(-jitter, jitter)))
        x = min(max(x, 0), max_x)
        y = min(max(y, 0), max_y)

    alpha = mask_image.filter(ImageFilter.GaussianBlur(feather_radius)) if feather_radius else mask_image
    composite = background.copy()
    composite.paste(patch, (x, y), alpha)

    pasted_mask = np.zeros((background.height, background.width), dtype=bool)
    pasted_mask[y : y + patch.height, x : x + patch.width] = np.asarray(mask_image) > 127
    return composite, pasted_mask


def synthesize_defect(
    background: Image.Image,
    source_image: Image.Image,
    source_mask: np.ndarray,
    rng: np.random.Generator,
    preserve_location: bool = True,
    **paste_kwargs,
) -> Optional[Tuple[Image.Image, np.ndarray]]:
    """Extract the defect from `source_image` and paste it onto `background`.

    `preserve_location` anchors the paste at the defect's original centre so
    it lands on the same part of the object it came from.
    """
    extracted = extract_defect_patch(source_image, source_mask)
    if extracted is None:
        return None
    patch, patch_mask = extracted
    if preserve_location and "anchor" not in paste_kwargs:
        scale_x = background.width / source_image.width
        scale_y = background.height / source_image.height
        centroid = mask_centroid(np.asarray(source_mask) > 0)
        if centroid is not None:
            paste_kwargs["anchor"] = (centroid[0] * scale_x, centroid[1] * scale_y)
    return paste_defect(background, patch, patch_mask, rng, **paste_kwargs)
