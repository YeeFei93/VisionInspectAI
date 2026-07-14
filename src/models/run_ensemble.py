"""Hybrid ensemble: fuse the supervised baseline classifier's prediction with
the unsupervised PatchCore anomaly score into one combined decision.

Evaluated only on the classifier's held-out validation subset (the same
split produced by `make_train_val_split`) — that's the only data the
classifier hasn't been fit on, so it's the fair, leakage-free ground for
comparing the classifier alone, PatchCore alone, and the fused ensemble.

Usage:
    python -m src.models.run_ensemble --config config/screw_config.yaml
    python -m src.models.run_ensemble --config config/screw_config.yaml \
        --classifier-architecture efficientnet_b0 --classifier-weight 0.4
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from sklearn.metrics import roc_auc_score

from src.data.dataset import load_manifest, make_train_val_split
from src.evaluation.metrics import compute_classification_metrics, youden_threshold
from src.models.anomaly_detector import PatchCoreAnomalyDetector
from src.models.baseline_classifier import build_baseline_model
from src.preprocessing.segmentation import compute_foreground_mask
from src.preprocessing.transform import get_val_transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/screw_config.yaml"))
    parser.add_argument(
        "--classifier-architecture",
        default="resnet18",
        help="Which trained baseline_<arch>_<category>.pt checkpoint to use as the classifier half.",
    )
    parser.add_argument(
        "--classifier-weight",
        type=float,
        default=0.5,
        help="Fusion weight for the classifier probability (PatchCore gets 1 - this).",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    category = config.get("category", "screw")
    data_cfg = config["data"]
    model_cfg = config["model"]
    anomaly_cfg = config["anomaly_detection"]
    output_cfg = config["output"]

    manifest = load_manifest(PROJECT_ROOT / data_cfg["manifest_path"])
    _train_subset, val_subset = make_train_val_split(
        manifest, val_split=data_cfg["val_split"], seed=data_cfg["seed"]
    )
    print(f"Evaluating ensemble on the classifier's held-out val subset: {len(val_subset)} images")

    image_size = data_cfg["image_size"]
    transform = get_val_transforms(image_size)
    device = torch.device("cpu")

    # --- Classifier half ---
    classifier_arch = args.classifier_architecture
    classifier = build_baseline_model(
        architecture=classifier_arch, num_classes=model_cfg["num_classes"], pretrained=False
    )
    classifier_ckpt = PROJECT_ROOT / output_cfg["checkpoint_dir"] / f"baseline_{classifier_arch}_{category}.pt"
    classifier.load_state_dict(torch.load(classifier_ckpt, map_location=device))
    classifier.eval()

    # --- PatchCore half ---
    detector = PatchCoreAnomalyDetector(
        backbone=anomaly_cfg["backbone"], layers=tuple(anomaly_cfg["layers"]), device="cpu"
    )
    detector_ckpt = (
        PROJECT_ROOT / output_cfg["checkpoint_dir"] / f"patchcore_{anomaly_cfg['backbone']}_{category}_memory_bank.pt"
    )
    detector.load(detector_ckpt)

    patchcore_metrics = json.loads(
        (PROJECT_ROOT / output_cfg["metrics_dir"] / f"patchcore_{anomaly_cfg['backbone']}_{category}_metrics.json")
        .read_text()
    )
    patchcore_score_min = patchcore_metrics["score_min"]
    patchcore_score_max = patchcore_metrics["score_max"]

    labels, classifier_probs, patchcore_scores = [], [], []

    with torch.no_grad():
        for _, row in val_subset.iterrows():
            image = Image.open(PROJECT_ROOT / row["image_path"]).convert("RGB")
            resized_image = image.resize((image_size, image_size))
            input_tensor = transform(image).unsqueeze(0)

            logits = classifier(input_tensor)
            prob_defective = F.softmax(logits, dim=1)[0, 1].item()
            classifier_probs.append(prob_defective)

            foreground_mask = torch.from_numpy(compute_foreground_mask(resized_image, image_size)).unsqueeze(0)
            result = detector.predict(input_tensor, foreground_masks=foreground_mask)[0]
            patchcore_scores.append(result.image_score)

            labels.append(int(row["label"]))

    labels_arr = np.array(labels)
    classifier_probs_arr = np.array(classifier_probs)
    patchcore_scores_arr = np.array(patchcore_scores)

    # Rescale PatchCore's raw distance score to a comparable [0, 1] "probability of defective".
    patchcore_norm = np.clip(
        (patchcore_scores_arr - patchcore_score_min) / max(patchcore_score_max - patchcore_score_min, 1e-6), 0, 1
    )

    w = args.classifier_weight
    fused_scores = w * classifier_probs_arr + (1 - w) * patchcore_norm

    results = {}
    for name, scores in [
        ("classifier_only", classifier_probs_arr),
        ("patchcore_only", patchcore_norm),
        ("fused_ensemble", fused_scores),
    ]:
        threshold = youden_threshold(labels_arr, scores)
        preds = (scores >= threshold).astype(int)
        metrics = compute_classification_metrics(labels_arr, preds)
        metrics["auroc"] = float(roc_auc_score(labels_arr, scores))
        metrics["threshold"] = float(threshold)
        results[name] = metrics
        print(f"\n=== {name} ===")
        print(json.dumps(metrics, indent=2))

    metrics_dir = PROJECT_ROOT / output_cfg["metrics_dir"]
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_path = metrics_dir / f"ensemble_{classifier_arch}_{category}_metrics.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved comparison metrics to {out_path}")


if __name__ == "__main__":
    main()
