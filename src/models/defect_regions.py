"""Per-region defect-type classification (Option A: connected-component
splitting of the anomaly mask + reusing the whole-image defect-type
classifier on each cropped region).

MVTec-AD's per-category defect_type label is one string per image, even for
its "combined" type (multiple defect phenomena in one image) -- the ground
truth mask only marks the union of all anomalous pixels, with no per-type
breakdown. Rather than requiring new multi-label annotations, this module
splits the *anomaly mask* into distinct connected regions and classifies
each crop independently with the existing single-type classifier, so an
image with several simultaneous defects can report more than one type.

Caveat: MVTec provides no ground truth for which specific types make up a
"combined" image, so this is an approximate, visually-checkable breakdown,
not something with a directly computable accuracy number.
"""

from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.preprocessing.transform import get_val_transforms


@dataclass
class DefectRegion:
    bbox: tuple  # (x, y, w, h) in the resized-image coordinate space
    area: int
    defect_type: str
    confidence: float


def find_defect_regions(binary_mask: np.ndarray, min_area: int = 30) -> list:
    """Connected components of a binary anomaly mask, filtered by pixel area."""
    mask_u8 = binary_mask.astype(np.uint8) * 255
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    boxes = []
    for label in range(1, num_labels):  # label 0 is the background component
        x, y, w, h, area = stats[label]
        if area >= min_area:
            boxes.append((int(x), int(y), int(w), int(h), int(area)))
    return boxes


def classify_defect_regions(
    resized_image: Image.Image,
    anomaly_map: torch.Tensor,
    threshold: float,
    defect_model,
    defect_types: list,
    padding: int = 8,
    min_area: int = 30,
) -> list:
    """Crop each connected anomalous region (padded) out of `resized_image`
    and run the whole-image defect-type classifier on each crop separately."""
    anomaly_map_np = anomaly_map.detach().cpu().numpy()
    binary_mask = anomaly_map_np >= threshold
    image_size = resized_image.size[0]
    transform = get_val_transforms(image_size)

    regions = []
    for x, y, w, h, area in find_defect_regions(binary_mask, min_area=min_area):
        x0, y0 = max(0, x - padding), max(0, y - padding)
        x1, y1 = min(image_size, x + w + padding), min(image_size, y + h + padding)
        crop = resized_image.crop((x0, y0, x1, y1)).resize((image_size, image_size))

        input_tensor = transform(crop).unsqueeze(0)
        with torch.no_grad():
            probs = F.softmax(defect_model(input_tensor), dim=1)[0]
        pred_idx = int(probs.argmax().item())

        regions.append(
            DefectRegion(
                bbox=(x, y, w, h),
                area=area,
                defect_type=defect_types[pred_idx],
                confidence=float(probs[pred_idx].item()),
            )
        )
    return regions


def summarize_distinct_types(regions: list) -> list:
    """Collapse per-region predictions into one entry per distinct defect
    type, keeping the highest-confidence region for each type."""
    best_by_type = {}
    for region in regions:
        current = best_by_type.get(region.defect_type)
        if current is None or region.confidence > current.confidence:
            best_by_type[region.defect_type] = region
    return sorted(best_by_type.values(), key=lambda r: r.confidence, reverse=True)
