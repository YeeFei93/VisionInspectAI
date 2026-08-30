"""Evaluate a focused defect classifier with deployment-time PatchCore crops.

Usage:
    python -m src.evaluation.evaluate_focused_defect_classifier \
        --config config/screw_config.yaml
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from src.models.anomaly_detector import PatchCoreAnomalyDetector, scoring_artifact_suffix
from src.models.baseline_classifier import build_baseline_model
from src.models.train_defect_classifier import build_defect_manifest
from src.preprocessing.defect_crop import crop_to_defect
from src.preprocessing.segmentation import compute_foreground_mask
from src.preprocessing.transform import get_val_transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--crop-source",
        choices=("ground-truth", "patchcore"),
        default="ground-truth",
        help="Focused classifier variant to evaluate with deployment-time PatchCore crops.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open() as config_file:
        config = yaml.safe_load(config_file)

    category = config["category"]
    data_cfg = config["data"]
    model_cfg = config["model"]
    anomaly_cfg = config["anomaly_detection"]
    output_cfg = config["output"]
    manifest, defect_types = build_defect_manifest(PROJECT_ROOT / data_cfg["manifest_path"])
    _, val_subset = train_test_split(
        manifest,
        test_size=data_cfg["val_split"],
        random_state=data_cfg["seed"],
        stratify=manifest["label"],
    )

    run_name = f"defect_classifier_{model_cfg['architecture']}_{category}_focused"
    if args.crop_source == "patchcore":
        run_name += "_patchcore"
    checkpoint_dir = PROJECT_ROOT / output_cfg["checkpoint_dir"]
    metrics_dir = PROJECT_ROOT / output_cfg["metrics_dir"]
    classifier_path = checkpoint_dir / f"{run_name}.pt"
    classifier_metrics_path = metrics_dir / f"{run_name}_metrics.json"
    detector_name = f"patchcore_{anomaly_cfg['backbone']}_{category}"
    detector_path = checkpoint_dir / f"{detector_name}_memory_bank.pt"
    detector_metrics_name = detector_name + scoring_artifact_suffix(
        anomaly_cfg.get("num_neighbors", 1),
        anomaly_cfg.get("softmax_reweighting", False),
        anomaly_cfg.get("reweight_num_neighbors", 9),
    )
    detector_metrics_path = metrics_dir / f"{detector_metrics_name}_metrics.json"
    for path in (
        classifier_path,
        classifier_metrics_path,
        detector_path,
        detector_metrics_path,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing required artifact: {path}")

    classifier_metrics = json.loads(classifier_metrics_path.read_text())
    detector_metrics = json.loads(detector_metrics_path.read_text())
    pixel_threshold = float(detector_metrics["pixel_threshold"])
    image_size = data_cfg["image_size"]
    transform = get_val_transforms(image_size)

    classifier = build_baseline_model(
        architecture=model_cfg["architecture"],
        num_classes=len(defect_types),
        pretrained=False,
    )
    classifier.load_state_dict(torch.load(classifier_path, map_location="cpu"))
    classifier.eval()
    detector = PatchCoreAnomalyDetector(
        backbone=anomaly_cfg["backbone"],
        layers=tuple(anomaly_cfg["layers"]),
        device="cpu",
        num_neighbors=anomaly_cfg.get("num_neighbors", 1),
        softmax_reweighting=anomaly_cfg.get("softmax_reweighting", False),
        reweight_num_neighbors=anomaly_cfg.get("reweight_num_neighbors", 9),
    )
    detector.load(detector_path)

    use_foreground_mask = anomaly_cfg.get("use_foreground_mask", True)
    y_true, y_pred = [], []
    for _, row in val_subset.iterrows():
        image = Image.open(PROJECT_ROOT / row["image_path"]).convert("RGB")
        resized_image = image.resize((image_size, image_size))
        input_tensor = transform(image).unsqueeze(0)
        if use_foreground_mask:
            foreground_mask = compute_foreground_mask(resized_image, image_size)
            foreground_tensor = torch.from_numpy(foreground_mask).unsqueeze(0)
        else:
            foreground_mask = np.ones((image_size, image_size), dtype=bool)
            foreground_tensor = None

        result = detector.predict(input_tensor, foreground_masks=foreground_tensor)[0]
        predicted_mask = np.logical_and(
            result.anomaly_map.numpy() >= pixel_threshold,
            foreground_mask,
        )
        focused_image = crop_to_defect(
            image,
            predicted_mask,
            padding_ratio=classifier_metrics["crop_padding_ratio"],
            min_crop_fraction=classifier_metrics["min_crop_fraction"],
        )
        with torch.no_grad():
            prediction = classifier(transform(focused_image).unsqueeze(0)).argmax(dim=1).item()
        y_true.append(int(row["label"]))
        y_pred.append(int(prediction))

    accuracy = float(accuracy_score(y_true, y_pred))
    classifier_metrics["predicted_crop_val_accuracy"] = accuracy
    classifier_metrics["predicted_crop_confusion_matrix"] = confusion_matrix(
        y_true, y_pred, labels=list(range(len(defect_types)))
    ).tolist()
    classifier_metrics["predicted_crop_source"] = "patchcore_pixel_threshold"
    classifier_metrics_path.write_text(json.dumps(classifier_metrics, indent=2))
    print(f"{category} predicted-crop validation accuracy: {accuracy:.4f}")
    print(f"Updated {classifier_metrics_path}")


if __name__ == "__main__":
    main()