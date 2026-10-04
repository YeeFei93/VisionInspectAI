from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from src.inference.inspection_pipeline import (
    InspectionPipeline,
    LowCategoryConfidenceError,
)
from src.models.defect_regions import keep_primary_anomaly_region


class FakeDetector:
    def __init__(self, score: float):
        self.score = score

    def predict(self, input_tensor, foreground_masks=None):
        image_size = input_tensor.shape[-1]
        return [
            SimpleNamespace(
                image_score=self.score,
                anomaly_map=torch.full((image_size, image_size), self.score),
            )
        ]


class FakeInspectionPipeline(InspectionPipeline):
    def __init__(self, category_confidence: float = 0.95):
        super().__init__()
        self.category_confidence = category_confidence

    def detect_category(self, image):
        return "screw", self.category_confidence

    def _load_config(self, category):
        return {
            "data": {"image_size": 16},
            "anomaly_detection": {"use_foreground_mask": False},
        }

    def _load_detector(self, category, config):
        return FakeDetector(score=0.25)

    def _load_metrics(self, category, config):
        return {"threshold": 0.5, "pixel_threshold": 0.1}


def test_inspect_uses_pixel_threshold_for_localization_visuals(monkeypatch):
    captured_threshold = None

    def fake_make_overlay(image, anomaly_map, threshold):
        nonlocal captured_threshold
        captured_threshold = threshold
        image_size = anomaly_map.shape[-1]
        pixels = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        return np.zeros((image_size, image_size)), pixels, pixels

    monkeypatch.setattr(
        "src.inference.inspection_pipeline.make_overlay", fake_make_overlay
    )
    result = FakeInspectionPipeline().inspect(Image.new("RGB", (24, 24), "white"))

    assert result.category == "screw"
    assert result.detected_category == "screw"
    assert result.category_confidence == pytest.approx(0.95)
    assert result.prediction == "Normal"
    assert result.anomaly_score == pytest.approx(0.25)
    assert result.threshold == pytest.approx(0.5)
    assert result.pixel_threshold == pytest.approx(0.1)
    assert result.severity == "High"
    assert captured_threshold == pytest.approx(0.1)
    assert result.resized_image.size == (16, 16)
    assert result.anomaly_map.shape == (16, 16)
    assert isinstance(result.heatmap, np.ndarray)
    assert isinstance(result.overlay, np.ndarray)


def test_low_category_confidence_blocks_automatic_inspection():
    pipeline = FakeInspectionPipeline(category_confidence=0.59)

    with pytest.raises(LowCategoryConfidenceError):
        pipeline.inspect(Image.new("RGB", (24, 24), "white"))


def test_manual_category_allows_inspection_when_detection_confidence_is_low():
    pipeline = FakeInspectionPipeline(category_confidence=0.59)

    result = pipeline.inspect(
        Image.new("RGB", (24, 24), "white"), category="bottle"
    )

    assert result.detected_category == "screw"
    assert result.category == "bottle"
    assert result.prediction == "Normal"


def test_default_pipeline_routes_screw_to_high_resolution_config():
    pipeline = InspectionPipeline()

    assert pipeline.category_configs["screw"].name == "screw_config_highres_patchcore.yaml"


def test_keep_primary_anomaly_region_suppresses_disconnected_noise():
    anomaly_map = torch.zeros((8, 8))
    anomaly_map[1:3, 1:3] = 0.7
    anomaly_map[5:7, 5:7] = 0.8
    anomaly_map[6, 6] = 1.0

    filtered = keep_primary_anomaly_region(anomaly_map, threshold=0.5)

    assert torch.all(filtered[1:3, 1:3] == 0)
    assert torch.equal(filtered[5:7, 5:7], anomaly_map[5:7, 5:7])


def test_keep_primary_anomaly_region_leaves_subthreshold_map_unchanged():
    anomaly_map = torch.full((4, 4), 0.2)

    filtered = keep_primary_anomaly_region(anomaly_map, threshold=0.5)

    assert torch.equal(filtered, anomaly_map)


def test_keep_primary_anomaly_region_accepts_display_fill_value():
    anomaly_map = torch.zeros((8, 8))
    anomaly_map[1:3, 1:3] = 0.7
    anomaly_map[5:7, 5:7] = 0.8
    anomaly_map[6, 6] = 1.0

    filtered = keep_primary_anomaly_region(
        anomaly_map, threshold=0.5, suppressed_value=0.49
    )

    assert torch.allclose(
        filtered[1:3, 1:3], torch.full((2, 2), 0.49)
    )
    assert torch.equal(filtered[5:7, 5:7], anomaly_map[5:7, 5:7])


def test_keep_primary_anomaly_region_uses_peak_relative_growth_boundary():
    anomaly_map = torch.zeros((5, 5))
    anomaly_map[1:4, 1:4] = 0.6
    anomaly_map[2, 2] = 1.0

    filtered = keep_primary_anomaly_region(
        anomaly_map, threshold=0.5, peak_fraction=0.5
    )

    assert filtered[2, 2] == 1.0
    assert torch.all(filtered[1:4, 1:4][filtered[1:4, 1:4] != 1.0] == 0)


@pytest.mark.parametrize("peak_fraction", [-0.1, 1.0])
def test_keep_primary_anomaly_region_rejects_invalid_peak_fraction(peak_fraction):
    with pytest.raises(ValueError):
        keep_primary_anomaly_region(
            torch.ones((4, 4)), threshold=0.5, peak_fraction=peak_fraction
        )