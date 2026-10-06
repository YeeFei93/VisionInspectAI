import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.data.dataset import ManifestImageDataset
from src.models.defect_regions import classify_defect_regions
from src.preprocessing.defect_crop import crop_to_defect, defect_crop_box


def test_defect_crop_box_is_square_and_keeps_edge_defect_inside():
    mask = np.zeros((100, 120), dtype=bool)
    mask[0:10, 110:120] = True

    box = defect_crop_box(mask, padding_ratio=0.25, min_crop_fraction=0.2)

    assert box is not None
    left, top, right, bottom = box
    assert right - left == bottom - top
    assert left <= 110 and right >= 120
    assert top == 0 and bottom >= 10


def test_crop_to_defect_resizes_mask_coordinates_and_crops_image():
    image = Image.new("RGB", (200, 100), "white")
    mask = np.zeros((50, 100), dtype=bool)
    mask[20:30, 45:55] = True

    cropped = crop_to_defect(image, mask, padding_ratio=0, min_crop_fraction=0.2)

    assert cropped.size == (20, 20)


def test_crop_to_defect_returns_full_image_for_empty_mask():
    image = Image.new("RGB", (80, 60), "white")

    cropped = crop_to_defect(image, np.zeros((60, 80), dtype=bool))

    assert cropped is image


def test_manifest_dataset_applies_defect_mask_crop_before_transform(tmp_path):
    image = Image.new("RGB", (100, 100), "white")
    image.save(tmp_path / "image.png")
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, 45:55] = 255
    Image.fromarray(mask).save(tmp_path / "mask.png")
    manifest = pd.DataFrame(
        [{"image_path": "image.png", "mask_path": "mask.png", "label": 2}]
    )
    dataset = ManifestImageDataset(
        manifest,
        tmp_path,
        transform=lambda cropped: cropped.size,
        focus_on_mask=True,
        crop_padding_ratio=0,
        min_crop_fraction=0.2,
    )

    crop_size, label = dataset[0]

    assert crop_size == (20, 20)
    assert label == 2


def test_region_classifier_uses_classifier_image_size():
    class ShapeRecordingModel(torch.nn.Module):
        def forward(self, images):
            self.input_shape = tuple(images.shape)
            return torch.tensor([[0.9, 0.1]])

    image = Image.new("RGB", (32, 32), "white")
    anomaly_map = torch.zeros((32, 32))
    anomaly_map[8:14, 8:14] = 0.8
    model = ShapeRecordingModel()

    regions = classify_defect_regions(
        image,
        anomaly_map,
        threshold=0.5,
        defect_model=model,
        defect_types=["scratch", "dent"],
        padding=0,
        min_area=1,
        classifier_image_size=16,
    )

    assert len(regions) == 1
    assert model.input_shape == (1, 3, 16, 16)


@pytest.mark.parametrize(
    ("padding_ratio", "min_crop_fraction"),
    [(-0.1, 0.25), (0.25, 0), (0.25, 1.1)],
)
def test_defect_crop_box_rejects_invalid_settings(padding_ratio, min_crop_fraction):
    with pytest.raises(ValueError):
        defect_crop_box(np.ones((10, 10)), padding_ratio, min_crop_fraction)