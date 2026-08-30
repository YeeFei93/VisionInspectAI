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
    plot_four_way_distribution,
    plot_generalization_gap,
    plot_memory_bank_pca,
    plot_score_distribution,
    plot_threshold_sweep,
    youden_threshold,
)
from src.models.anomaly_detector import PatchCoreAnomalyDetector, scoring_artifact_suffix
from src.preprocessing.segmentation import compute_foreground_mask
from src.preprocessing.transform import get_patchcore_bank_augmentation_transforms, get_val_transforms
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
    parser.add_argument(
        "--generalization-holdout-ratio",
        type=float,
        default=0.0,
        help=(
            "Reserve this fraction of train/good (excluded from the memory bank) to check "
            "whether the bank generalizes to unseen normal images (0 disables the check)."
        ),
    )
    parser.add_argument(
        "--threshold-strategy",
        choices=("youden", "holdout-margin", "percentile"),
        default="youden",
        help=(
            "'youden' (default) picks the threshold that maximizes TPR-FPR on the labeled test "
            "set. 'holdout-margin' instead calibrates it from normal-only scores (memory-bank + "
            "held-out train/good, via --generalization-holdout-ratio) as mean + "
            "--threshold-margin-std standard deviations -- a safety margin against normal "
            "variation the test set's Youden threshold doesn't account for. 'percentile' "
            "instead uses the --threshold-percentile percentile of those same normal-only "
            "scores, avoiding holdout-margin's Gaussian assumption. Both require "
            "--generalization-holdout-ratio > 0."
        ),
    )
    parser.add_argument(
        "--threshold-margin-std",
        type=float,
        default=3.0,
        help="Standard deviations above the normal-score mean for --threshold-strategy holdout-margin.",
    )
    parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=95.0,
        help="Percentile (0-100] of normal-only scores used for --threshold-strategy percentile.",
    )
    parser.add_argument(
        "--coreset-ratio",
        type=float,
        default=None,
        help="Override anomaly_detection.coreset_ratio (fraction of all patches eligible for the coreset).",
    )
    parser.add_argument(
        "--bank-augmentation",
        action="store_true",
        help=(
            "Apply mild brightness/contrast/rotation/translation/scale augmentation to train/good "
            "before feature extraction, to expand the normal manifold the memory bank is built from."
        ),
    )
    parser.add_argument(
        "--bank-augmentation-passes",
        type=int,
        default=3,
        help="Number of independently-augmented passes over train/good when --bank-augmentation is set.",
    )
    parser.add_argument(
        "--perturbation-confidence",
        action="store_true",
        help=(
            "For held-out train/good images (requires --generalization-holdout-ratio > 0), also "
            "re-score each image under mild brightness/contrast perturbations and report the score's "
            "standard deviation as a per-image stability/confidence signal -- independent of the mean "
            "generalization gap, an image whose score swings widely under mild perturbation is sitting "
            "in an unstable region of the memory bank's scoring function."
        ),
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


class RepeatedDataset(torch.utils.data.Dataset):
    """Repeats a dataset `repeats` times so a stochastic transform is
    re-sampled fresh each repetition -- used to draw multiple independently
    augmented views per image when building the PatchCore memory bank."""

    def __init__(self, dataset, repeats: int):
        self.dataset = dataset
        self.repeats = repeats

    def __len__(self):
        return len(self.dataset) * self.repeats

    def __getitem__(self, idx):
        return self.dataset[idx % len(self.dataset)]


def score_image_rows(
    rows,
    detector: PatchCoreAnomalyDetector,
    transform,
    image_size: int,
    use_foreground_mask: bool,
) -> tuple:
    """Run the detector over manifest rows and return (scores, image_paths).
    Used for the held-out normal-image generalization check; the paths let
    us identify which specific held-out image is hardest."""
    scores = []
    paths = []
    for _, row in rows.iterrows():
        image = Image.open(PROJECT_ROOT / row["image_path"]).convert("RGB")
        resized_image = image.resize((image_size, image_size))
        input_tensor = transform(image).unsqueeze(0)

        foreground_mask = None
        if use_foreground_mask:
            foreground_mask = torch.from_numpy(compute_foreground_mask(resized_image, image_size)).unsqueeze(0)

        result = detector.predict(input_tensor, foreground_masks=foreground_mask)[0]
        scores.append(result.image_score)
        paths.append(row["image_path"])
    return scores, paths


def score_image_rows_with_perturbation_confidence(
    rows,
    detector: PatchCoreAnomalyDetector,
    transform,
    image_size: int,
    use_foreground_mask: bool,
) -> tuple:
    """Like score_image_rows, but also re-scores each image under mild
    brightness/contrast perturbations and returns the per-image standard
    deviation across those views as a stability/confidence signal: a
    normal image whose score swings widely under a perturbation this small
    is sitting in an unstable region of the memory bank's scoring
    function, independent of its (possibly unremarkable) mean score."""
    import torchvision.transforms.functional as TF

    scores = []
    paths = []
    perturbation_stds = []
    for _, row in rows.iterrows():
        image = Image.open(PROJECT_ROOT / row["image_path"]).convert("RGB")
        resized_image = image.resize((image_size, image_size))

        foreground_mask = None
        if use_foreground_mask:
            foreground_mask = torch.from_numpy(compute_foreground_mask(resized_image, image_size)).unsqueeze(0)

        variants = [
            image,
            TF.adjust_brightness(image, 1.15),
            TF.adjust_brightness(image, 0.85),
            TF.adjust_contrast(image, 1.15),
        ]
        variant_scores = []
        for variant in variants:
            input_tensor = transform(variant).unsqueeze(0)
            result = detector.predict(input_tensor, foreground_masks=foreground_mask)[0]
            variant_scores.append(result.image_score)

        scores.append(variant_scores[0])  # unperturbed score, same as score_image_rows
        paths.append(row["image_path"])
        perturbation_stds.append(float(np.std(variant_scores)))
    return scores, paths, perturbation_stds


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
    coreset_ratio = (
        args.coreset_ratio if args.coreset_ratio is not None else anomaly_cfg["coreset_ratio"]
    )
    if max_coreset_size <= 0:
        raise ValueError("max_coreset_size must be positive")
    if projection_dim <= 0:
        raise ValueError("projection_dim must be positive")
    if num_neighbors <= 0:
        raise ValueError("num_neighbors must be positive")
    if reweight_num_neighbors <= 1:
        raise ValueError("reweight_num_neighbors must be greater than 1")
    if not 0 < coreset_ratio <= 1:
        raise ValueError("coreset_ratio must be between 0 (exclusive) and 1 (inclusive)")
    if args.threshold_strategy in ("holdout-margin", "percentile") and args.generalization_holdout_ratio <= 0:
        raise ValueError(
            f"--threshold-strategy {args.threshold_strategy} requires --generalization-holdout-ratio > 0"
        )
    if not 0 < args.threshold_percentile <= 100:
        raise ValueError("threshold_percentile must be between 0 (exclusive) and 100 (inclusive)")
    if args.perturbation_confidence and args.generalization_holdout_ratio <= 0:
        raise ValueError("--perturbation-confidence requires --generalization-holdout-ratio > 0")
    print(
        f"Coreset projection: {projection_method} | Seed: {seed} | "
        f"Max coreset: {max_coreset_size} | Coreset ratio: {coreset_ratio} | "
        f"Projection dim: {projection_dim} | "
        f"Layers: {', '.join(layers)} | k-NN: {num_neighbors} | "
        f"Softmax reweighting: {softmax_reweighting}"
    )

    generalization_holdout_ratio = args.generalization_holdout_ratio
    holdout_rows = train_rows.iloc[0:0]
    bank_rows = train_rows
    if generalization_holdout_ratio > 0:
        if not 0 < generalization_holdout_ratio < 1:
            raise ValueError("generalization_holdout_ratio must be between 0 and 1 (exclusive)")
        shuffled_rows = train_rows.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        num_holdout = max(1, int(len(shuffled_rows) * generalization_holdout_ratio))
        holdout_rows = shuffled_rows.iloc[:num_holdout].reset_index(drop=True)
        bank_rows = shuffled_rows.iloc[num_holdout:].reset_index(drop=True)
        print(
            f"Generalization holdout: reserving {len(holdout_rows)}/{len(train_rows)} "
            f"train/good images (fitting the memory bank on the remaining {len(bank_rows)})"
        )

    train_dataset = ManifestImageDataset(bank_rows, PROJECT_ROOT, transform=transform)
    if args.bank_augmentation:
        augmentation_transform = get_patchcore_bank_augmentation_transforms(image_size)
        augmented_dataset = ManifestImageDataset(bank_rows, PROJECT_ROOT, transform=augmentation_transform)
        train_dataset = RepeatedDataset(augmented_dataset, repeats=args.bank_augmentation_passes)
        print(
            f"Bank augmentation enabled: {args.bank_augmentation_passes} independently-augmented "
            f"passes over {len(bank_rows)} train/good images ({len(train_dataset)} total feature-extraction views)"
        )
    train_loader = DataLoader(train_dataset, batch_size=anomaly_cfg["batch_size"], shuffle=False)

    detector = PatchCoreAnomalyDetector(
        backbone=anomaly_cfg["backbone"],
        layers=layers,
        coreset_ratio=coreset_ratio,
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
    youden_threshold_value = youden_threshold(labels_arr, scores_arr)

    # Generalization check: does the memory bank generalize to normal images it
    # never saw, or does it only recognize the specific train/good images it was
    # built from? A held-out score distribution shifted well above the
    # memory-bank distribution indicates the latter (overfitting). Computed
    # before threshold selection so --threshold-strategy holdout-margin can use it.
    bank_scores = holdout_scores = None
    perturbation_stds = None
    if generalization_holdout_ratio > 0:
        print(f"Scoring {len(holdout_rows)} held-out train/good images for the generalization check...")
        holdout_scores, holdout_paths = score_image_rows(holdout_rows, detector, transform, image_size, use_foreground_mask)
        bank_scores, _bank_paths = score_image_rows(bank_rows, detector, transform, image_size, use_foreground_mask)
        hardest_order = np.argsort(holdout_scores)[::-1][:3]
        print("Hardest held-out train/good images (highest anomaly score):")
        for rank in hardest_order:
            print(f"  {holdout_scores[rank]:.4f}  {holdout_paths[rank]}")

        if args.perturbation_confidence:
            print(f"Re-scoring {len(holdout_rows)} held-out images under mild perturbations for confidence check...")
            _pconf_scores, pconf_paths, perturbation_stds = score_image_rows_with_perturbation_confidence(
                holdout_rows, detector, transform, image_size, use_foreground_mask
            )
            least_stable_order = np.argsort(perturbation_stds)[::-1][:3]
            print("Least-stable held-out images (highest score std under perturbation):")
            for rank in least_stable_order:
                print(
                    f"  score={holdout_scores[rank]:.4f}  perturbation_std={perturbation_stds[rank]:.4f}  "
                    f"{pconf_paths[rank]}"
                )

    if args.threshold_strategy == "holdout-margin":
        # Calibrated from normal-only scores (never the labeled test set), so it
        # carries an explicit safety margin instead of the Youden threshold's
        # arbitrary pick within a perfectly-separable test set's good/defective gap.
        normal_scores = np.array(bank_scores + holdout_scores)
        threshold = float(normal_scores.mean() + args.threshold_margin_std * normal_scores.std())
    elif args.threshold_strategy == "percentile":
        # Also calibrated from normal-only scores, but as a percentile rather than
        # mean + k*std -- doesn't assume the normal-score distribution is Gaussian.
        normal_scores = np.array(bank_scores + holdout_scores)
        threshold = float(np.percentile(normal_scores, args.threshold_percentile))
    else:
        threshold = youden_threshold_value

    predictions = (scores_arr >= threshold).astype(int)
    metrics = compute_classification_metrics(labels_arr, predictions)
    metrics["auroc"] = float(auroc)
    metrics["threshold"] = float(threshold)
    metrics["threshold_strategy"] = args.threshold_strategy
    metrics["youden_threshold"] = float(youden_threshold_value)
    if args.threshold_strategy == "holdout-margin":
        metrics["threshold_margin_std"] = args.threshold_margin_std
    if args.threshold_strategy == "percentile":
        metrics["threshold_percentile"] = args.threshold_percentile
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
    print(f"Chosen threshold ({args.threshold_strategy}): {threshold:.4f}")
    if args.threshold_strategy != "youden":
        print(f"(Youden's J threshold would have been: {youden_threshold_value:.4f})")

    if generalization_holdout_ratio > 0:
        generalization_gap = float(np.mean(holdout_scores) - np.mean(bank_scores))
        metrics["generalization_holdout_ratio"] = generalization_holdout_ratio
        metrics["memory_bank_score_mean"] = float(np.mean(bank_scores))
        metrics["memory_bank_score_max"] = float(np.max(bank_scores))
        metrics["holdout_score_mean"] = float(np.mean(holdout_scores))
        metrics["holdout_score_max"] = float(np.max(holdout_scores))
        metrics["generalization_gap"] = generalization_gap
        print(
            f"Memory-bank train/good mean score: {np.mean(bank_scores):.4f} | "
            f"Held-out train/good mean score: {np.mean(holdout_scores):.4f} | "
            f"Generalization gap: {generalization_gap:.4f}"
        )
        print(
            f"Held-out max score: {np.max(holdout_scores):.4f} vs. decision threshold: {threshold:.4f} "
            f"(margin: {threshold - np.max(holdout_scores):.4f})"
        )
        if perturbation_stds is not None:
            metrics["holdout_perturbation_std_mean"] = float(np.mean(perturbation_stds))
            metrics["holdout_perturbation_std_max"] = float(np.max(perturbation_stds))
            print(
                f"Held-out perturbation-confidence std: mean {np.mean(perturbation_stds):.4f} | "
                f"max {np.max(perturbation_stds):.4f}"
            )

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
    figures_dir = PROJECT_ROOT / output_cfg.get("figures_dir", "outputs/figures")
    for directory in (checkpoint_dir, metrics_dir, heatmaps_dir, figures_dir):
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
    if generalization_holdout_ratio > 0:
        # Fits the bank on less than the full train/good set -- keep this out of the
        # canonical run name so it never overwrites the full-data checkpoint/metrics.
        run_name += f"_genholdout{generalization_holdout_ratio}"
    if args.coreset_ratio is not None:
        run_name += f"_cr{args.coreset_ratio}"
    if args.bank_augmentation:
        run_name += f"_bankaug{args.bank_augmentation_passes}"
    if args.threshold_strategy != "youden":
        # Same reasoning: a non-default threshold changes the reported classification
        # metrics, so it must never collide with the canonical Youden-threshold run.
        run_name += f"_thr{args.threshold_strategy}"
        if args.threshold_strategy == "holdout-margin" and args.threshold_margin_std != 3.0:
            run_name += f"_std{args.threshold_margin_std}"
        if args.threshold_strategy == "percentile" and args.threshold_percentile != 95.0:
            run_name += f"_pct{args.threshold_percentile}"
    if args.perturbation_confidence:
        run_name += "_pconf"
    if not args.metrics_only:
        detector.save(checkpoint_dir / f"{run_name}_memory_bank.pt")
    (metrics_dir / f"{run_name}_metrics.json").write_text(json.dumps(metrics, indent=2))

    # PatchCore fits in one shot (no epochs), so these stand in for a
    # per-epoch loss/accuracy curve: how well the scores separate
    # good/defective, and how accuracy/precision/recall/F1 trade off
    # across candidate thresholds around the chosen one.
    title_prefix = f"PatchCore ({category})"
    plot_memory_bank_pca(
        detector.memory_bank.numpy(),
        output_path=figures_dir / f"{run_name}_memory_bank_pca.png",
        title_prefix=title_prefix,
    )
    plot_score_distribution(
        scores_arr,
        labels_arr,
        threshold,
        output_path=figures_dir / f"{run_name}_score_distribution.png",
        title_prefix=title_prefix,
    )
    plot_threshold_sweep(
        scores_arr,
        labels_arr,
        threshold,
        output_path=figures_dir / f"{run_name}_threshold_sweep.png",
        title_prefix=title_prefix,
    )
    if holdout_scores is not None:
        plot_generalization_gap(
            bank_scores,
            holdout_scores,
            output_path=figures_dir / f"{run_name}_generalization_gap.png",
            title_prefix=title_prefix,
        )
        plot_four_way_distribution(
            bank_scores,
            holdout_scores,
            scores_arr[labels_arr == 0].tolist(),
            scores_arr[labels_arr == 1].tolist(),
            threshold,
            output_path=figures_dir / f"{run_name}_four_way_distribution.png",
            title_prefix=title_prefix,
        )

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
    print(f"Saved score distribution + threshold sweep plots to {figures_dir}")


if __name__ == "__main__":
    main()
