"""Framework-independent image inspection orchestration."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from src.models.anomaly_detector import PatchCoreAnomalyDetector, scoring_artifact_suffix
from src.models.baseline_classifier import build_baseline_model
from src.models.defect_regions import (
    classify_defect_regions,
    keep_primary_anomaly_region,
    summarize_distinct_types,
)
from src.preprocessing.defect_crop import crop_to_defect
from src.preprocessing.segmentation import compute_foreground_mask
from src.preprocessing.transform import get_val_transforms
from src.visualization.heatmap import make_overlay


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CATEGORY_CLASSIFIER_IMAGE_SIZE = 224
CONFIDENCE_THRESHOLD = 0.60
DEFAULT_CATEGORY_CONFIGS = {
    category: PROJECT_ROOT / "config" / f"{category}_config.yaml"
    for category in (
        "screw",
        "bottle",
        "hazelnut",
        "carpet",
        "leather",
        "transistor",
        "grid",
        "tile",
        "wood",
    )
}


@dataclass(frozen=True)
class DefectPrediction:
    defect_type: str
    confidence: float
    area: int | None = None


@dataclass
class InspectionResult:
    category: str
    category_confidence: float
    detected_category: str
    prediction: str
    anomaly_score: float
    threshold: float
    pixel_threshold: float
    severity: str | None
    severity_reason: str
    defect_types: list[DefectPrediction] = field(default_factory=list)
    distinct_defect_regions: list[DefectPrediction] = field(default_factory=list)
    defect_model_available: bool = False
    resized_image: Image.Image | None = None
    anomaly_map: np.ndarray | None = None
    heatmap: np.ndarray | None = None
    overlay: np.ndarray | None = None


class InspectionSetupError(RuntimeError):
    """Raised when a required trained artifact is unavailable."""


class LowCategoryConfidenceError(RuntimeError):
    """Raised when automatic category routing is not reliable enough."""

    def __init__(self, category: str, confidence: float):
        self.category = category
        self.confidence = confidence
        super().__init__(
            f"Category confidence {confidence:.0%} is below the "
            f"{CONFIDENCE_THRESHOLD:.0%} inspection threshold."
        )


def should_make_prediction(
    confidence: float, threshold: float = CONFIDENCE_THRESHOLD
) -> bool:
    return confidence >= threshold


def classify_severity(
    anomaly_map: torch.Tensor, foreground_mask: np.ndarray, threshold: float
) -> tuple[str | None, str]:
    anomaly_map_np = anomaly_map.detach().cpu().numpy()
    foreground_area = int(foreground_mask.sum())
    if foreground_area == 0:
        return None, "No object detected in the image."

    anomalous_area = int(
        np.logical_and(anomaly_map_np >= threshold, foreground_mask).sum()
    )
    fraction = anomalous_area / foreground_area

    if fraction < 0.05:
        return "Low", "Anomaly area is small and localized."
    if fraction < 0.20:
        return "Medium", "Anomaly covers a moderate portion of the object."
    return "High", "Anomaly covers a large portion of the object."


class InspectionPipeline:
    def __init__(
        self,
        project_root: Path = PROJECT_ROOT,
        category_configs: dict[str, Path] | None = None,
        device: str = "cpu",
    ) -> None:
        self.project_root = project_root
        self.category_configs = category_configs or {
            category: project_root
            / "config"
            / (
                "screw_config_highres_patchcore.yaml"
                if category == "screw"
                else f"{category}_config.yaml"
            )
            for category in DEFAULT_CATEGORY_CONFIGS
        }
        self.device = device
        self._category_classifier: tuple[Any, list[str]] | None = None
        self._configs: dict[str, dict] = {}
        self._detectors: dict[str, PatchCoreAnomalyDetector] = {}
        self._metrics: dict[str, dict] = {}
        self._defect_classifiers: dict[str, tuple[Any, list[str], dict] | None] = {}

    @property
    def known_categories(self) -> list[str]:
        _, categories = self._load_category_classifier()
        return categories

    def inspect(
        self, image: Image.Image, category: str | None = None
    ) -> InspectionResult:
        detected_category, category_confidence = self.detect_category(image)
        if category is None:
            if not should_make_prediction(category_confidence):
                raise LowCategoryConfidenceError(
                    detected_category, category_confidence
                )
            category = detected_category
        if category not in self.category_configs:
            raise ValueError(f"Unsupported category: {category}")

        config = self._load_config(category)
        detector = self._load_detector(category, config)
        metrics = self._load_metrics(category, config)
        threshold = float(metrics["threshold"])
        pixel_threshold = float(metrics.get("pixel_threshold", threshold))
        image_size = int(config["data"]["image_size"])
        transform = get_val_transforms(image_size)
        resized_image = image.resize((image_size, image_size))
        input_tensor = transform(image).unsqueeze(0)

        if config["anomaly_detection"].get("use_foreground_mask", True):
            foreground_mask = compute_foreground_mask(resized_image, image_size)
            foreground_mask_tensor = torch.from_numpy(foreground_mask).unsqueeze(0)
        else:
            foreground_mask = np.ones((image_size, image_size), dtype=bool)
            foreground_mask_tensor = None

        anomaly_result = detector.predict(
            input_tensor, foreground_masks=foreground_mask_tensor
        )[0]
        anomaly_score = float(anomaly_result.image_score)
        prediction = "Defective" if anomaly_score >= threshold else "Normal"
        localization_map = anomaly_result.anomaly_map
        if config["anomaly_detection"].get("keep_primary_region", False):
            localization_map = keep_primary_anomaly_region(
                localization_map,
                pixel_threshold,
                peak_fraction=config["anomaly_detection"].get(
                    "primary_region_peak_fraction", 0.0
                ),
            )
        severity, severity_reason = classify_severity(
            localization_map, foreground_mask, pixel_threshold
        )

        defect_types: list[DefectPrediction] = []
        distinct_regions: list[DefectPrediction] = []
        defect_model_available = False
        if prediction == "Defective":
            defect_classifier = self._load_defect_classifier(category, config)
            defect_model_available = defect_classifier is not None
            if defect_classifier is not None:
                defect_model, type_names, metadata = defect_classifier
                defect_input = input_tensor
                if metadata.get("crop_mode") == "defect_focused":
                    predicted_mask = np.logical_and(
                        localization_map.detach().cpu().numpy() >= pixel_threshold,
                        foreground_mask,
                    )
                    defect_image = crop_to_defect(
                        image,
                        predicted_mask,
                        padding_ratio=metadata.get("crop_padding_ratio", 0.25),
                        min_crop_fraction=metadata.get("min_crop_fraction", 0.25),
                    )
                    defect_input = transform(defect_image).unsqueeze(0)

                defect_type, confidence = self._detect_defect_type(
                    defect_model, type_names, defect_input
                )
                if should_make_prediction(confidence):
                    defect_types.append(
                        DefectPrediction(defect_type, confidence)
                    )

                regions = classify_defect_regions(
                    resized_image,
                    localization_map,
                    pixel_threshold,
                    defect_model,
                    type_names,
                )
                distinct_regions = [
                    DefectPrediction(
                        region.defect_type, region.confidence, region.area
                    )
                    for region in summarize_distinct_types(regions)
                    if should_make_prediction(region.confidence)
                ]

        display_map = localization_map
        if config["anomaly_detection"].get("keep_primary_region", False):
            display_map = keep_primary_anomaly_region(
                anomaly_result.anomaly_map,
                pixel_threshold,
                suppressed_value=float(
                    np.nextafter(
                        np.float32(pixel_threshold), np.float32("-inf")
                    )
                ),
                peak_fraction=config["anomaly_detection"].get(
                    "primary_region_peak_fraction", 0.0
                ),
            )
        _, heatmap, overlay = make_overlay(
            resized_image,
            display_map,
            threshold=pixel_threshold,
        )
        return InspectionResult(
            category=category,
            category_confidence=category_confidence,
            detected_category=detected_category,
            prediction=prediction,
            anomaly_score=anomaly_score,
            threshold=threshold,
            pixel_threshold=pixel_threshold,
            severity=severity,
            severity_reason=severity_reason,
            defect_types=defect_types,
            distinct_defect_regions=distinct_regions,
            defect_model_available=defect_model_available,
            resized_image=resized_image,
            anomaly_map=localization_map.detach().cpu().numpy(),
            heatmap=heatmap,
            overlay=overlay,
        )

    def detect_category(self, image: Image.Image) -> tuple[str, float]:
        model, categories = self._load_category_classifier()
        transform = get_val_transforms(CATEGORY_CLASSIFIER_IMAGE_SIZE)
        input_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            probabilities = F.softmax(model(input_tensor), dim=1)[0]
        prediction_index = int(probabilities.argmax().item())
        return (
            categories[prediction_index],
            float(probabilities[prediction_index].item()),
        )

    def _load_category_classifier(self) -> tuple[Any, list[str]]:
        if self._category_classifier is not None:
            return self._category_classifier

        checkpoint_path = (
            self.project_root
            / "models"
            / "checkpoints"
            / "category_classifier_resnet18.pt"
        )
        metrics_path = (
            self.project_root
            / "outputs"
            / "metrics"
            / "category_classifier_metrics.json"
        )
        if not checkpoint_path.exists() or not metrics_path.exists():
            raise InspectionSetupError(
                "No category classifier found. Run "
                "`python -m src.models.train_category_classifier` first."
            )

        with metrics_path.open() as metrics_file:
            categories = json.load(metrics_file)["categories"]
        model = build_baseline_model(
            architecture="resnet18",
            num_classes=len(categories),
            pretrained=False,
        )
        model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        model.eval()
        self._category_classifier = model, categories
        return self._category_classifier

    def _load_config(self, category: str) -> dict:
        if category not in self._configs:
            with self.category_configs[category].open() as config_file:
                self._configs[category] = yaml.safe_load(config_file)
        return self._configs[category]

    def _load_detector(
        self, category: str, config: dict
    ) -> PatchCoreAnomalyDetector:
        if category in self._detectors:
            return self._detectors[category]

        anomaly_config = config["anomaly_detection"]
        run_name = anomaly_config.get(
            "artifact_name", f"patchcore_{anomaly_config['backbone']}_{category}"
        )
        checkpoint_path = (
            self.project_root
            / config["output"]["checkpoint_dir"]
            / f"{run_name}_memory_bank.pt"
        )
        if not checkpoint_path.exists():
            raise InspectionSetupError(
                f"No trained memory bank found at {checkpoint_path}. Run "
                f"`python -m src.models.run_anomaly_detection --config "
                f"config/{category}_config.yaml` first."
            )

        detector = PatchCoreAnomalyDetector(
            backbone=anomaly_config["backbone"],
            layers=tuple(anomaly_config["layers"]),
            device=self.device,
            num_neighbors=anomaly_config.get("num_neighbors", 1),
            softmax_reweighting=anomaly_config.get(
                "softmax_reweighting", False
            ),
            reweight_num_neighbors=anomaly_config.get(
                "reweight_num_neighbors", 9
            ),
        )
        detector.load(checkpoint_path)
        self._detectors[category] = detector
        return detector

    def _load_metrics(self, category: str, config: dict) -> dict:
        if category in self._metrics:
            return self._metrics[category]

        anomaly_config = config["anomaly_detection"]
        run_name = anomaly_config.get("artifact_name")
        if run_name is None:
            run_name = f"patchcore_{anomaly_config['backbone']}_{category}"
            run_name += scoring_artifact_suffix(
                anomaly_config.get("num_neighbors", 1),
                anomaly_config.get("softmax_reweighting", False),
                anomaly_config.get("reweight_num_neighbors", 9),
            )
        metrics_path = (
            self.project_root
            / config["output"]["metrics_dir"]
            / f"{run_name}_metrics.json"
        )
        if not metrics_path.exists():
            raise InspectionSetupError(
                f"No metrics file found at {metrics_path}. Run "
                f"`python -m src.models.run_anomaly_detection --config "
                f"config/{category}_config.yaml` first."
            )

        with metrics_path.open() as metrics_file:
            self._metrics[category] = json.load(metrics_file)
        return self._metrics[category]

    def _load_defect_classifier(
        self, category: str, config: dict
    ) -> tuple[Any, list[str], dict] | None:
        if category in self._defect_classifiers:
            return self._defect_classifiers[category]

        model_config = config["model"]
        base_name = (
            f"defect_classifier_{model_config['architecture']}_{category}"
        )
        checkpoint_dir = self.project_root / config["output"]["checkpoint_dir"]
        metrics_dir = self.project_root / config["output"]["metrics_dir"]
        focused_config = config.get("defect_classifier", {})
        use_focused_crops = focused_config.get("use_focused_crops", False)
        run_name = base_name
        if use_focused_crops:
            run_name += "_focused"
            if focused_config.get("crop_source") == "patchcore":
                run_name += "_patchcore"
        checkpoint_path = checkpoint_dir / f"{run_name}.pt"
        metrics_path = metrics_dir / f"{run_name}_metrics.json"

        if use_focused_crops and (
            not checkpoint_path.exists() or not metrics_path.exists()
        ):
            checkpoint_path = checkpoint_dir / f"{base_name}.pt"
            metrics_path = metrics_dir / f"{base_name}_metrics.json"

        if not checkpoint_path.exists() or not metrics_path.exists():
            self._defect_classifiers[category] = None
            return None

        with metrics_path.open() as metrics_file:
            metadata = json.load(metrics_file)
        defect_types = metadata["defect_types"]
        model = build_baseline_model(
            architecture=model_config["architecture"],
            num_classes=len(defect_types),
            pretrained=False,
        )
        model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        model.eval()
        classifier = model, defect_types, metadata
        self._defect_classifiers[category] = classifier
        return classifier

    @staticmethod
    def _detect_defect_type(
        model: Any, defect_types: list[str], input_tensor: torch.Tensor
    ) -> tuple[str, float]:
        with torch.no_grad():
            probabilities = F.softmax(model(input_tensor), dim=1)[0]
        prediction_index = int(probabilities.argmax().item())
        return (
            defect_types[prediction_index],
            float(probabilities[prediction_index].item()),
        )