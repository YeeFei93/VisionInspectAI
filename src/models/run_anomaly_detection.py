"""Train and evaluate the main PatchCore anomaly detection model.

Trained only on train/good (no labels needed); evaluated on the full
test/ split (good + every defect type). For each test image this produces
an image-level anomaly score, a good/defective prediction, and a heatmap
highlighting the suspected defect region.

Usage:
    python -m src.models.run_anomaly_detection --config config/screw_config.yaml
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.data.dataset import ManifestImageDataset, load_manifest
from src.evaluation.metrics import (
    compute_classification_metrics,
    compute_pixel_level_metrics,
    youden_threshold,
)
from src.models.anomaly_detector import PatchCoreAnomalyDetector
from src.preprocessing.transform import get_val_transforms
from src.visualization.heatmap import save_anomaly_heatmap

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/screw_config.yaml"))
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def load_gt_mask(mask_path, image_size: int) -> np.ndarray:
    """Load a ground-truth defect mask (or an all-zero mask for good images)
    and resize it to match the anomaly map resolution."""
    if not mask_path:
        return np.zeros((image_size, image_size), dtype=np.uint8)
    mask_img = Image.open(mask_path).convert("L").resize((image_size, image_size), Image.NEAREST)
    return (np.array(mask_img) > 127).astype(np.uint8)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    category = config.get("category", "screw")
    data_cfg = config["data"]
    anomaly_cfg = config["anomaly_detection"]
    output_cfg = config["output"]

    manifest = load_manifest(PROJECT_ROOT / data_cfg["manifest_path"])
    train_rows = manifest[(manifest["split"] == "train") & (manifest["label"] == 0)].reset_index(drop=True)
    test_rows = manifest[manifest["split"] == "test"].reset_index(drop=True)

    image_size = data_cfg["image_size"]
    transform = get_val_transforms(image_size)

    train_dataset = ManifestImageDataset(train_rows, PROJECT_ROOT, transform=transform)
    test_dataset = ManifestImageDataset(test_rows, PROJECT_ROOT, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=anomaly_cfg["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=anomaly_cfg["batch_size"], shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = PatchCoreAnomalyDetector(
        backbone=anomaly_cfg["backbone"],
        layers=tuple(anomaly_cfg["layers"]),
        coreset_ratio=anomaly_cfg["coreset_ratio"],
        max_coreset_size=anomaly_cfg["max_coreset_size"],
        projection_dim=anomaly_cfg["projection_dim"],
        device=device,
    )

    print(f"Fitting PatchCore memory bank on {len(train_dataset)} train/good images...")
    detector.fit(train_loader)
    print(f"Memory bank size: {detector.memory_bank.shape[0]} patches")

    print(f"Scoring {len(test_dataset)} test images...")
    scores, labels, anomaly_maps = [], [], []
    for images, batch_labels in test_loader:
        for result in detector.predict(images):
            scores.append(result.image_score)
            anomaly_maps.append(result.anomaly_map)
        labels.extend(batch_labels.tolist())

    scores_arr = np.array(scores)
    labels_arr = np.array(labels)

    auroc = roc_auc_score(labels_arr, scores_arr)
    threshold = youden_threshold(labels_arr, scores_arr)
    predictions = (scores_arr >= threshold).astype(int)
    metrics = compute_classification_metrics(labels_arr, predictions)
    metrics["auroc"] = float(auroc)
    metrics["threshold"] = float(threshold)
    metrics["score_min"] = float(scores_arr.min())
    metrics["score_max"] = float(scores_arr.max())

    print(f"Image-level ROC-AUC: {auroc:.4f}")
    print(f"Chosen threshold (Youden's J): {threshold:.4f}")

    # Pixel-level localization: predicted heatmap vs ground_truth mask.
    print("Loading ground-truth masks for pixel-level evaluation...")
    gt_masks = [
        load_gt_mask(
            PROJECT_ROOT / row["mask_path"] if isinstance(row["mask_path"], str) and row["mask_path"] else None,
            image_size,
        )
        for _, row in test_rows.iterrows()
    ]
    anomaly_maps_np = [m.numpy() for m in anomaly_maps]
    pixel_metrics = compute_pixel_level_metrics(anomaly_maps_np, gt_masks)
    metrics.update(pixel_metrics)

    print(f"Pixel-level ROC-AUC: {pixel_metrics['pixel_auroc']:.4f}")
    print(f"Mean IoU: {pixel_metrics['mean_iou']:.4f}")
    print(f"Mean Dice: {pixel_metrics['mean_dice']:.4f}")
    print(json.dumps(metrics, indent=2))

    checkpoint_dir = PROJECT_ROOT / output_cfg["checkpoint_dir"]
    metrics_dir = PROJECT_ROOT / output_cfg["metrics_dir"]
    heatmaps_dir = PROJECT_ROOT / output_cfg.get("heatmaps_dir", "outputs/heatmaps")
    for directory in (checkpoint_dir, metrics_dir, heatmaps_dir):
        directory.mkdir(parents=True, exist_ok=True)

    run_name = f"patchcore_{anomaly_cfg['backbone']}_{category}"
    detector.save(checkpoint_dir / f"{run_name}_memory_bank.pt")
    (metrics_dir / f"{run_name}_metrics.json").write_text(json.dumps(metrics, indent=2))

    # Save one example heatmap per defect type (plus "good") for a qualitative check.
    vmin, vmax = float(scores_arr.min()), float(scores_arr.max())
    seen_defect_types = set()
    saved_examples = []
    for idx, row in test_rows.iterrows():
        defect_type = row["defect_type"]
        if defect_type in seen_defect_types:
            continue
        seen_defect_types.add(defect_type)

        image = Image.open(PROJECT_ROOT / row["image_path"]).convert("RGB").resize((image_size, image_size))
        score = scores[idx]
        prediction = "Defective" if score >= threshold else "Good"
        out_path = heatmaps_dir / f"{run_name}_{defect_type}_example.png"
        example_gt_mask = gt_masks[idx] if defect_type != "good" else None
        save_anomaly_heatmap(
            image, anomaly_maps[idx], out_path, score=score, vmin=vmin, vmax=vmax, gt_mask=example_gt_mask
        )
        saved_examples.append(out_path)
        print(f"[{defect_type}] Anomaly score: {score:.2f} | Prediction: {prediction} | Heatmap: {out_path}")

    print(f"Saved {len(saved_examples)} example heatmaps to {heatmaps_dir}")
    print(f"Saved memory bank to {checkpoint_dir / f'{run_name}_memory_bank.pt'}")
    print(f"Saved metrics to {metrics_dir / f'{run_name}_metrics.json'}")


if __name__ == "__main__":
    main()
