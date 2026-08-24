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
from src.models.anomaly_detector import PatchCoreAnomalyDetector, scoring_artifact_suffix
from src.preprocessing.segmentation import compute_foreground_mask
from src.preprocessing.transform import get_val_transforms
from src.visualization.heatmap import save_anomaly_heatmap

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAX_CORESET_SIZE = 2000
DEFAULT_PROJECTION_DIM = 128
DEFAULT_LAYERS = ("layer2", "layer3")
DEFAULT_NUM_NEIGHBORS = 1
DEFAULT_REWEIGHT_NUM_NEIGHBORS = 9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/screw_config.yaml"))
    parser.add_argument(
        "--projection-method",
        choices=("random", "pca"),
        default=None,
        help="Override anomaly_detection.projection_method from the config.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the coreset/PCA seed and include it in the artifact name.",
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Save metrics but skip the memory-bank checkpoint and example heatmaps.",
    )
    parser.add_argument(
        "--max-coreset-size",
        type=int,
        default=None,
        help="Override anomaly_detection.max_coreset_size.",
    )
    parser.add_argument(
        "--projection-dim",
        type=int,
        default=None,
        help="Override anomaly_detection.projection_dim.",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        choices=("layer1", "layer2", "layer3", "layer4"),
        default=None,
        help="Override anomaly_detection.layers, for example --layers layer2 layer3.",
    )
    parser.add_argument(
        "--num-neighbors",
        type=int,
        default=None,
        help="Average this many nearest memory distances per patch (default: 1).",
    )
    parser.add_argument(
        "--softmax-reweighting",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable PatchCore neighborhood softmax image-score reweighting.",
    )
    parser.add_argument(
        "--reweight-num-neighbors",
        type=int,
        default=None,
        help="Memory neighborhood size for softmax reweighting (default: 9).",
    )
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


def build_run_name(
    backbone: str,
    category: str,
    projection_method: str,
    seed: int,
    args: argparse.Namespace,
    include_seed: bool = False,
    max_coreset_size: int = DEFAULT_MAX_CORESET_SIZE,
    projection_dim: int = DEFAULT_PROJECTION_DIM,
    layers: tuple = DEFAULT_LAYERS,
    num_neighbors: int = DEFAULT_NUM_NEIGHBORS,
    softmax_reweighting: bool = False,
    reweight_num_neighbors: int = DEFAULT_REWEIGHT_NUM_NEIGHBORS,
) -> str:
    effective_coreset_size = (
        args.max_coreset_size if args.max_coreset_size is not None else max_coreset_size
    )
    effective_projection_dim = (
        args.projection_dim if args.projection_dim is not None else projection_dim
    )
    effective_layers = tuple(args.layers) if args.layers is not None else layers
    run_name = f"patchcore_{backbone}_{category}"
    if projection_method != "random":
        run_name += f"_{projection_method}proj"
    if args.seed is not None or include_seed:
        run_name += f"_seed{seed}"
    if effective_coreset_size != DEFAULT_MAX_CORESET_SIZE:
        run_name += f"_cs{effective_coreset_size}"
    if effective_projection_dim != DEFAULT_PROJECTION_DIM:
        run_name += f"_pd{effective_projection_dim}"
    if effective_layers != DEFAULT_LAYERS:
        run_name += f"_{'-'.join(effective_layers)}"
    run_name += scoring_artifact_suffix(
        num_neighbors, softmax_reweighting, reweight_num_neighbors
    )
    return run_name


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
    train_loader = DataLoader(train_dataset, batch_size=anomaly_cfg["batch_size"], shuffle=False)

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")
    projection_method = args.projection_method or anomaly_cfg.get("projection_method", "random")
    seed = args.seed if args.seed is not None else anomaly_cfg.get("seed", data_cfg["seed"])
    max_coreset_size = (
        args.max_coreset_size
        if args.max_coreset_size is not None
        else anomaly_cfg["max_coreset_size"]
    )
    projection_dim = (
        args.projection_dim
        if args.projection_dim is not None
        else anomaly_cfg["projection_dim"]
    )
    layers = tuple(args.layers or anomaly_cfg["layers"])
    num_neighbors = (
        args.num_neighbors
        if args.num_neighbors is not None
        else anomaly_cfg.get("num_neighbors", DEFAULT_NUM_NEIGHBORS)
    )
    softmax_reweighting = (
        args.softmax_reweighting
        if args.softmax_reweighting is not None
        else anomaly_cfg.get("softmax_reweighting", False)
    )
    reweight_num_neighbors = (
        args.reweight_num_neighbors
        if args.reweight_num_neighbors is not None
        else anomaly_cfg.get(
            "reweight_num_neighbors", DEFAULT_REWEIGHT_NUM_NEIGHBORS
        )
    )
    if max_coreset_size <= 0:
        raise ValueError("max_coreset_size must be positive")
    if projection_dim <= 0:
        raise ValueError("projection_dim must be positive")
    if num_neighbors <= 0:
        raise ValueError("num_neighbors must be positive")
    if reweight_num_neighbors <= 1:
        raise ValueError("reweight_num_neighbors must be greater than 1")
    print(
        f"Coreset projection: {projection_method} | Seed: {seed} | "
        f"Max coreset: {max_coreset_size} | Projection dim: {projection_dim} | "
        f"Layers: {', '.join(layers)} | k-NN: {num_neighbors} | "
        f"Softmax reweighting: {softmax_reweighting}"
    )
    detector = PatchCoreAnomalyDetector(
        backbone=anomaly_cfg["backbone"],
        layers=layers,
        coreset_ratio=anomaly_cfg["coreset_ratio"],
        max_coreset_size=max_coreset_size,
        projection_dim=projection_dim,
        device=device,
        seed=seed,
        projection_method=projection_method,
        num_neighbors=num_neighbors,
        softmax_reweighting=softmax_reweighting,
        reweight_num_neighbors=reweight_num_neighbors,
    )

    print(f"Fitting PatchCore memory bank on {len(train_dataset)} train/good images...")
    detector.fit(train_loader)
    print(f"Memory bank size: {detector.memory_bank.shape[0]} patches")

    use_foreground_mask = anomaly_cfg.get("use_foreground_mask", True)
    mask_note = "background masked out via foreground segmentation" if use_foreground_mask else "foreground masking disabled"
    print(f"Scoring {len(test_rows)} test images ({mask_note})...")
    scores, labels, anomaly_maps = [], [], []
    for _, row in test_rows.iterrows():
        image = Image.open(PROJECT_ROOT / row["image_path"]).convert("RGB")
        resized_image = image.resize((image_size, image_size))
        input_tensor = transform(image).unsqueeze(0)

        foreground_mask = None
        if use_foreground_mask:
            foreground_mask = torch.from_numpy(compute_foreground_mask(resized_image, image_size)).unsqueeze(0)

        result = detector.predict(input_tensor, foreground_masks=foreground_mask)[0]
        scores.append(result.image_score)
        anomaly_maps.append(result.anomaly_map)
        labels.append(int(row["label"]))

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
    metrics["projection_method"] = projection_method
    metrics["seed"] = seed
    metrics["max_coreset_size"] = max_coreset_size
    metrics["memory_bank_size"] = int(detector.memory_bank.shape[0])
    metrics["projection_dim"] = projection_dim
    metrics["layers"] = list(layers)
    metrics["num_neighbors"] = num_neighbors
    metrics["softmax_reweighting"] = softmax_reweighting
    metrics["reweight_num_neighbors"] = reweight_num_neighbors

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

    run_name = build_run_name(
        anomaly_cfg["backbone"],
        category,
        projection_method,
        seed,
        args,
        include_seed="seed" in anomaly_cfg,
        max_coreset_size=max_coreset_size,
        projection_dim=projection_dim,
        layers=layers,
        num_neighbors=num_neighbors,
        softmax_reweighting=softmax_reweighting,
        reweight_num_neighbors=reweight_num_neighbors,
    )
    if not args.metrics_only:
        detector.save(checkpoint_dir / f"{run_name}_memory_bank.pt")
    (metrics_dir / f"{run_name}_metrics.json").write_text(json.dumps(metrics, indent=2))

    if args.metrics_only:
        print(f"Saved metrics to {metrics_dir / f'{run_name}_metrics.json'}")
        return

    # Save one example heatmap per defect type (plus "good") for a qualitative check.
    # Colors are anchored to the same image-level threshold used for the printed
    # Prediction below, so the heatmap's warm/cool split visually matches the
    # reported decision. vmin/vmax use the actual per-pixel value range (not the
    # image-level score range).
    vmin = float(min(m.min().item() for m in anomaly_maps))
    vmax = float(max(m.max().item() for m in anomaly_maps))
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
            image,
            anomaly_maps[idx],
            out_path,
            score=score,
            vmin=vmin,
            vmax=vmax,
            gt_mask=example_gt_mask,
            threshold=threshold,
        )
        saved_examples.append(out_path)
        print(f"[{defect_type}] Anomaly score: {score:.2f} | Prediction: {prediction} | Heatmap: {out_path}")

    print(f"Saved {len(saved_examples)} example heatmaps to {heatmaps_dir}")
    print(f"Saved memory bank to {checkpoint_dir / f'{run_name}_memory_bank.pt'}")
    print(f"Saved metrics to {metrics_dir / f'{run_name}_metrics.json'}")


if __name__ == "__main__":
    main()
