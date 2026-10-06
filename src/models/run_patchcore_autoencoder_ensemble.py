"""Fuse PatchCore (feature-space distance) and the convolutional autoencoder
(reconstruction error) into one unsupervised detector and compare all three
under the same protocol.

    S_ens = alpha * norm(S_PatchCore) + (1 - alpha) * norm(S_AE)    (image score)
    H_ens = alpha * norm(H_PatchCore) + (1 - alpha) * norm(H_AE)    (heatmap)

Protocol (no test-set tuning): both detectors use the same seeded
train/good -> fit / held-out-normal calibration split. PatchCore is re-fitted
on the fit subset and the autoencoder checkpoint (trained on the same fit
subset by `run_autoencoder.py`) is loaded. The per-detector normalizer and
the image/pixel thresholds (Nth percentile of the held-out normal scores)
come from the calibration images only; the labelled test set is scored once.
`--alpha` is fixed up front (default 0.5). The extra alpha / normalization
sweeps are reported as post-hoc sensitivity analysis and never feed back into
the primary result.

Usage:
    python -m src.models.run_patchcore_autoencoder_ensemble --config config/transistor_config.yaml
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.data.dataset import ManifestImageDataset, load_manifest
from src.evaluation.metrics import (
    compute_classification_metrics,
    compute_per_defect_type_detection,
    compute_pixel_level_metrics,
    percentile_threshold,
    plot_confusion_matrix,
    plot_score_distribution,
)
from src.models.anomaly_detector import PatchCoreAnomalyDetector
from src.models.anomaly_fusion import NORMALIZATION_METHODS, CalibrationNormalizer, fuse
from src.models.autoencoder import ERROR_METRICS, ConvAutoencoder
from src.models.defect_regions import keep_primary_anomaly_region
from src.models.run_anomaly_detection import (
    DEFAULT_CALIBRATION_HOLDOUT_RATIO,
    DEFAULT_CALIBRATION_PERCENTILE,
    DEFAULT_NUM_NEIGHBORS,
    DEFAULT_REWEIGHT_NUM_NEIGHBORS,
    load_config,
    load_gt_mask,
    split_calibration_rows,
)
from src.models.run_anomaly_detection import score_rows as score_patchcore
from src.models.run_autoencoder import get_device, scores_from_maps
from src.models.run_autoencoder import score_rows as score_autoencoder
from src.preprocessing.transform import get_autoencoder_transforms, get_val_transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SWEEP_ALPHAS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/transistor_config.yaml"))
    parser.add_argument("--alpha", type=float, default=0.5, help="PatchCore weight (fixed a priori).")
    parser.add_argument(
        "--losses",
        nargs="+",
        choices=ERROR_METRICS,
        default=list(ERROR_METRICS),
        help="Which trained autoencoder checkpoints to fuse with PatchCore.",
    )
    parser.add_argument(
        "--normalization",
        choices=NORMALIZATION_METHODS,
        default="zscore",
        help="Calibration-based normalizer for the primary result.",
    )
    parser.add_argument("--no-sweep", action="store_true", help="Skip the post-hoc alpha/normalization sweep.")
    return parser.parse_args()


def evaluate(
    cal_scores, cal_maps, test_scores, test_maps, labels, defect_types, gt_masks,
    percentile: float, keep_primary_region: bool, peak_fraction: float,
) -> dict:
    """Common evaluation: calibration-percentile thresholds, image metrics,
    pixel AUROC, and IoU/Dice raw and (optionally) after `keep_primary_region`."""
    cal_scores = np.asarray(cal_scores)
    test_scores = np.asarray(test_scores)
    image_threshold = percentile_threshold(cal_scores, percentile)
    pixel_threshold = percentile_threshold(
        np.concatenate([m.ravel() for m in cal_maps]), percentile
    )
    predictions = (test_scores >= image_threshold).astype(int)
    metrics = compute_classification_metrics(labels, predictions)
    metrics["auroc"] = float(roc_auc_score(labels, test_scores))
    metrics["threshold"] = float(image_threshold)
    metrics["per_defect_type"] = compute_per_defect_type_detection(
        defect_types, labels, test_scores, predictions
    )
    pixel = compute_pixel_level_metrics(test_maps, gt_masks, pixel_threshold=pixel_threshold)
    metrics["raw_mean_iou"], metrics["raw_mean_dice"] = pixel["mean_iou"], pixel["mean_dice"]
    metrics["pixel_auroc"], metrics["pixel_threshold"] = pixel["pixel_auroc"], pixel["pixel_threshold"]
    metrics["mean_iou"], metrics["mean_dice"] = pixel["mean_iou"], pixel["mean_dice"]
    if keep_primary_region:
        filtered_maps = [
            keep_primary_anomaly_region(
                torch.from_numpy(np.asarray(m, dtype=np.float32)), pixel_threshold, peak_fraction=peak_fraction
            ).numpy()
            for m in test_maps
        ]
        filtered = compute_pixel_level_metrics(filtered_maps, gt_masks, pixel_threshold=pixel_threshold)
        metrics["mean_iou"], metrics["mean_dice"] = filtered["mean_iou"], filtered["mean_dice"]
    return metrics


def normalize_detector(cal_scores, cal_maps, test_scores, test_maps, method: str):
    """Normalize one detector's image scores and heatmaps with statistics of
    the held-out normal calibration images."""
    score_norm = CalibrationNormalizer.fit(cal_scores, method)
    map_norm = CalibrationNormalizer.fit(np.concatenate([m.ravel() for m in cal_maps]), method)
    return (
        score_norm(cal_scores),
        [map_norm(m) for m in cal_maps],
        score_norm(test_scores),
        [map_norm(m) for m in test_maps],
    )


def fuse_detectors(pc_norm, ae_norm, alpha: float):
    cal_s = fuse(pc_norm[0], ae_norm[0], alpha)
    cal_m = [fuse(a, b, alpha) for a, b in zip(pc_norm[1], ae_norm[1])]
    test_s = fuse(pc_norm[2], ae_norm[2], alpha)
    test_m = [fuse(a, b, alpha) for a, b in zip(pc_norm[3], ae_norm[3])]
    return cal_s, cal_m, test_s, test_m


def save_comparison_figure(image, gt_mask, panels, output_path: Path, title: str) -> None:
    """Input + ground truth, then each detector's heatmap with the calibrated
    pixel threshold contour."""
    fig, axes = plt.subplots(1, len(panels) + 1, figsize=(3.6 * (len(panels) + 1), 3.8))
    axes[0].imshow(np.array(image))
    if gt_mask is not None and gt_mask.any():
        axes[0].contour(gt_mask, levels=[0.5], colors="lime", linewidths=1.2)
    axes[0].set_title(title, fontsize=9)
    for ax, (name, anomaly_map, threshold, vmax) in zip(axes[1:], panels):
        ax.imshow(anomaly_map, cmap="jet", vmin=0, vmax=vmax)
        if anomaly_map.max() >= threshold:
            ax.contour(anomaly_map, levels=[threshold], colors="white", linewidths=0.8)
        if gt_mask is not None and gt_mask.any():
            ax.contour(gt_mask, levels=[0.5], colors="lime", linewidths=0.8)
        ax.set_title(name, fontsize=9)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=110)
    plt.close(fig)


def summarize(metrics: dict) -> dict:
    keys = (
        "auroc", "pixel_auroc", "accuracy", "precision", "recall", "f1",
        "raw_mean_iou", "raw_mean_dice", "mean_iou", "mean_dice",
    )
    return {k: float(metrics[k]) for k in keys}


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    category = config.get("category", "transistor")
    data_cfg, anomaly_cfg, ae_cfg, output_cfg = (
        config["data"], config["anomaly_detection"], config["autoencoder"], config["output"],
    )
    image_size = data_cfg["image_size"]
    seed = anomaly_cfg.get("seed", data_cfg["seed"])
    calibration_cfg = anomaly_cfg.get("calibration", {})
    if not calibration_cfg.get("enabled", False):
        raise ValueError("Ensembling needs anomaly_detection.calibration.enabled: true (normal-only calibration)")
    holdout_ratio = calibration_cfg.get("holdout_ratio", DEFAULT_CALIBRATION_HOLDOUT_RATIO)
    percentile = calibration_cfg.get("percentile", DEFAULT_CALIBRATION_PERCENTILE)
    calibration_seed = calibration_cfg.get("seed", seed)
    use_foreground_mask = anomaly_cfg.get("use_foreground_mask", True)
    keep_region = anomaly_cfg.get("keep_primary_region", False)
    peak_fraction = anomaly_cfg.get("primary_region_peak_fraction", 0.0)

    manifest = load_manifest(PROJECT_ROOT / data_cfg["manifest_path"])
    train_rows = manifest[(manifest["split"] == "train") & (manifest["label"] == 0)].reset_index(drop=True)
    test_rows = manifest[manifest["split"] == "test"].reset_index(drop=True)
    fit_rows, calibration_rows = split_calibration_rows(train_rows, holdout_ratio, calibration_seed)
    print(f"Calibration split: {len(fit_rows)} fitting / {len(calibration_rows)} held-out normal images")

    labels = test_rows["label"].astype(int).to_numpy()
    defect_types = test_rows["defect_type"].tolist()
    gt_masks = [
        load_gt_mask(
            PROJECT_ROOT / row["mask_path"] if isinstance(row["mask_path"], str) and row["mask_path"] else None,
            image_size,
        )
        for _, row in test_rows.iterrows()
    ]

    device = get_device()
    torch.manual_seed(seed)
    pc_transform = get_val_transforms(image_size)
    detector = PatchCoreAnomalyDetector(
        backbone=anomaly_cfg["backbone"],
        layers=tuple(anomaly_cfg["layers"]),
        coreset_ratio=anomaly_cfg["coreset_ratio"],
        max_coreset_size=anomaly_cfg["max_coreset_size"],
        projection_dim=anomaly_cfg["projection_dim"],
        max_coreset_candidates=anomaly_cfg.get("max_coreset_candidates"),
        device=device,
        seed=seed,
        projection_method=anomaly_cfg.get("projection_method", "random"),
        num_neighbors=anomaly_cfg.get("num_neighbors", DEFAULT_NUM_NEIGHBORS),
        softmax_reweighting=anomaly_cfg.get("softmax_reweighting", False),
        reweight_num_neighbors=anomaly_cfg.get("reweight_num_neighbors", DEFAULT_REWEIGHT_NUM_NEIGHBORS),
    )
    print(f"Fitting PatchCore on {len(fit_rows)} train/good images (device: {device})...")
    detector.fit(
        DataLoader(
            ManifestImageDataset(fit_rows, PROJECT_ROOT, transform=pc_transform),
            batch_size=anomaly_cfg["batch_size"], shuffle=False,
        )
    )
    pc_cal_scores, pc_cal_maps = score_patchcore(calibration_rows, detector, pc_transform, image_size, use_foreground_mask)
    pc_test_scores, pc_test_maps = score_patchcore(test_rows, detector, pc_transform, image_size, use_foreground_mask)
    pc = (
        np.array(pc_cal_scores), [m.numpy() for m in pc_cal_maps],
        np.array(pc_test_scores), [m.numpy() for m in pc_test_maps],
    )

    eval_args = (labels, defect_types, gt_masks, percentile, keep_region, peak_fraction)
    results = {
        "protocol": {
            "alpha": args.alpha,
            "normalization": args.normalization,
            "normalizer_fitted_on": "held-out normal calibration images only",
            "threshold": f"{percentile}th percentile of calibration fused scores/pixels",
            "num_fit_images": len(fit_rows),
            "num_calibration_images": len(calibration_rows),
            "num_test_images": len(test_rows),
            "keep_primary_region": keep_region,
            "primary_region_peak_fraction": peak_fraction,
        },
        "patchcore": evaluate(*pc, *eval_args),
        "ensembles": {},
    }

    ae_transform = get_autoencoder_transforms(image_size)
    reduction = ae_cfg.get("score_reduction", "mean")
    topk_fraction = float(ae_cfg.get("topk_fraction", 0.01))
    blur_sigma = float(ae_cfg.get("error_blur_sigma", 4.0))
    figures_dir = PROJECT_ROOT / output_cfg.get("figures_dir", "outputs/figures")
    heatmaps_dir = PROJECT_ROOT / output_cfg.get("heatmaps_dir", "outputs/heatmaps")
    metrics_dir = PROJECT_ROOT / output_cfg["metrics_dir"]
    for directory in (figures_dir, heatmaps_dir, metrics_dir):
        directory.mkdir(parents=True, exist_ok=True)

    for loss in args.losses:
        checkpoint_path = PROJECT_ROOT / output_cfg["checkpoint_dir"] / f"autoencoder_conv_{loss}_{category}.pt"
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = ConvAutoencoder(
            image_size=checkpoint["image_size"],
            latent_dim=checkpoint["latent_dim"],
            base_channels=checkpoint["base_channels"],
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        score_args = (ae_transform, image_size, loss, blur_sigma, use_foreground_mask, device)
        cal_maps, _ = score_autoencoder(calibration_rows, model, *score_args)
        test_maps, _ = score_autoencoder(test_rows, model, *score_args)
        ae = (
            scores_from_maps(cal_maps, reduction, topk_fraction), [m.numpy() for m in cal_maps],
            scores_from_maps(test_maps, reduction, topk_fraction), [m.numpy() for m in test_maps],
        )

        entry = {"autoencoder": evaluate(*ae, *eval_args)}
        pc_norm = normalize_detector(*pc, args.normalization)
        ae_norm = normalize_detector(*ae, args.normalization)
        fused = fuse_detectors(pc_norm, ae_norm, args.alpha)
        entry["ensemble"] = evaluate(*fused, *eval_args)

        if not args.no_sweep:
            sweep = {}
            for method in NORMALIZATION_METHODS:
                p, a = normalize_detector(*pc, method), normalize_detector(*ae, method)
                sweep[method] = {
                    f"{alpha:.1f}": summarize(evaluate(*fuse_detectors(p, a, alpha), *eval_args))
                    for alpha in SWEEP_ALPHAS
                }
            entry["posthoc_sweep"] = sweep
        results["ensembles"][loss] = entry

        run_name = f"ensemble_patchcore_autoencoder_{loss}_{category}"
        ens = entry["ensemble"]
        predictions = (fused[2] >= ens["threshold"]).astype(int)
        plot_confusion_matrix(
            labels, predictions, output_path=figures_dir / f"{run_name}_confusion_matrix.png",
            title=f"PatchCore + AE-{loss.upper()} ensemble (α={args.alpha}) — confusion matrix",
        )
        plot_score_distribution(
            fused[2], labels, threshold=ens["threshold"],
            output_path=figures_dir / f"{run_name}_score_distribution.png",
            title=f"PatchCore + AE-{loss.upper()} ensemble (α={args.alpha}) — fused score",
        )

        pc_n, ae_n = pc_norm[3], ae_norm[3]
        pc_thr, ae_thr, ens_thr = (
            percentile_threshold(np.concatenate([m.ravel() for m in maps]), percentile)
            for maps in (pc_norm[1], ae_norm[1], fused[1])
        )
        seen = set()
        for idx, row in test_rows.iterrows():
            defect_type = row["defect_type"]
            if defect_type in seen:
                continue
            seen.add(defect_type)
            image = Image.open(PROJECT_ROOT / row["image_path"]).convert("RGB").resize((image_size, image_size))
            vmax = max(float(m.max()) for m in (pc_n[idx], ae_n[idx], fused[3][idx]))
            save_comparison_figure(
                image, gt_masks[idx],
                [
                    ("PatchCore", pc_n[idx], pc_thr, vmax),
                    (f"AE-{loss.upper()}", ae_n[idx], ae_thr, vmax),
                    (f"Ensemble α={args.alpha}", fused[3][idx], ens_thr, vmax),
                ],
                heatmaps_dir / f"{run_name}_{defect_type}_comparison.png",
                title=defect_type,
            )

        print(f"\n=== PatchCore + AE-{loss.upper()} (alpha={args.alpha}, {args.normalization}) ===")
        for name, m in (("PatchCore", results["patchcore"]), ("Autoencoder", entry["autoencoder"]), ("Ensemble", ens)):
            print(
                f"{name:12s} imgAUROC {m['auroc']:.4f} pixAUROC {m['pixel_auroc']:.4f} "
                f"P {m['precision']:.3f} R {m['recall']:.3f} F1 {m['f1']:.3f} "
                f"IoU {m['mean_iou']:.3f}/{m['raw_mean_iou']:.3f} Dice {m['mean_dice']:.3f}/{m['raw_mean_dice']:.3f}"
            )

    metrics_path = metrics_dir / f"ensemble_patchcore_autoencoder_{category}_metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2))
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
