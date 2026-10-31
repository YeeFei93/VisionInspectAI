"""Lightweight foreground (object) segmentation for the plain-background
MVTec-AD categories. Used to keep the anomaly score and heatmap focused on
the actual part (e.g. the screw) instead of the plain background, as
recommended for industrial anomaly detection demos."""

import cv2
import numpy as np
from PIL import Image


def compute_foreground_mask(image: Image.Image, size: int) -> np.ndarray:
    """Otsu-threshold foreground mask at `size x size` resolution.

    Assumes the background is a plain, roughly-uniform surface and the
    object of interest (the screw) occupies a minority of the image area —
    true for the MVTec-AD screw category. Returns a boolean (size, size)
    array where True = foreground/object, False = background.
    """
    gray = np.array(image.convert("L").resize((size, size)))
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = binary > 0

    # Otsu just splits pixels into two classes; assume the smaller class is
    # the object of interest and flip if that assumption doesn't hold.
    if mask.mean() > 0.5:
        mask = ~mask

    return mask
