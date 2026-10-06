"""Train and evaluate a convolutional autoencoder as a second unsupervised
anomaly detector, side by side with PatchCore.

Same protocol as `run_anomaly_detection.py` so the two are directly
comparable: trained only on train/good (no labels), the same seeded
normal-only calibration split chooses the image/pixel thresholds (Nth
percentile of held-out normal scores), and the frozen model is scored once
on the full labelled test set with the same metric code (image/pixel
AUROC, IoU, Dice, optional `keep_primary_region` postprocessing).

Anomaly map = per-pixel reconstruction error (squared error for `l2`,
1 - SSIM for `ssim`); image score = mean/max/top-k of that map. All three
score reductions' AUROCs are reported; the configured one drives the
thresholded predictions.

Usage:
    python -m src.models.run_autoencoder --config config/transistor_config.yaml
    python -m src.models.run_autoencoder --config config/transistor_config.yaml --loss ssim
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from src.data.dataset import ManifestImageDataset, load_manifest
from src.evaluation.metrics import (
    compute_classification_metrics,
    compute_per_defect_type_detection,
    compute_pixel_level_metrics,
    percentile_threshold,
    plot_confusion_matrix,
    plot_metric_vs_threshold,
    plot_score_distribution,
    youden_threshold,
)
from src.models.autoencoder import (
    ERROR_METRICS,
    SCORE_REDUCTIONS,
    ConvAutoencoder,
    image_score,
    reconstruction_error_map,
    reconstruction_loss,
)
from src.models.defect_regions import keep_primary_anomaly_region
from src.models.run_anomaly_detection import (
    DEFAULT_CALIBRATION_ENABLED,
    DEFAULT_CALIBRATION_HOLDOUT_RATIO,
    DEFAULT_CALIBRATION_PERCENTILE,
    load_config,
    load_gt_mask,
    split_calibration_rows,
)
from src.preprocessing.segmentation import compute_foreground_mask
from src.preprocessing.transform import get_autoencoder_transforms
from src.visualization.heatmap import save_anomaly_heatmap

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/transistor_config.yaml"))
    parser.add_argument(
        "--loss",
        choices=ERROR_METRICS,
        default=None,
        help="Override autoencoder.loss (also used as the per-pixel error metric).",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override autoencoder.epochs.")
    parser.add_argument(
        "--latent-dim", type=int, default=None, help="Override autoencoder.latent_dim."
    )
    parser.add_argument(
        "--score-reduction",
        choices=SCORE_REDUCTIONS,
        default=None,
        help="Override autoencoder.score_reduction (drives thresholded predictions).",
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Save metrics/figures but skip the checkpoint and example heatmaps.",
    )
    return parser.parse_args()


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def build_run_name(category: str, loss: str, latent_dim: int, default_latent_dim: int) -> str:
    run_name = f"autoencoder_conv_{loss}_{category}"
    if latent_dim != default_latent_dim:
        run_name += f"_ld{latent_dim}"
    return run_name


def preload(rows, transform) -> TensorDataset:
    """Decode + resize every image once; there is no random augmentation, so
    re-reading the full-resolution PNGs every epoch would only cost time."""
    dataset = ManifestImageDataset(rows, PROJECT_ROOT, transform=transform)
    images, labels = zip(*(dataset[i] for i in range(len(dataset))))
    return TensorDataset(torch.stack(images), torch.tensor(labels))


@torch.no_grad()
def evaluate_loss(model, loader, loss_name: str, device: str) -> float:
    model.eval()
    total, count = 0.0, 0
    for images, _ in loader:
        images = images.to(device)
        total += reconstruction_loss(images, model(images), loss_name).item() * len(images)
        count += len(images)
    return total / max(count, 1)


def train_autoencoder(model, train_loader, val_loader, loss_name, epochs, lr, weight_decay, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        running, count = 0.0, 0
        for images, _ in train_loader:
            images = images.to(device)
            loss = reconstruction_loss(images, model(images), loss_name)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * len(images)
            count += len(images)
        entry = {"epoch": epoch, "train_loss": running / count}
        if val_loader is not None:
            entry["val_loss"] = evaluate_loss(model, val_loader, loss_name, device)
        history.append(entry)
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            val_note = f" | held-out normal loss {entry['val_loss']:.5f}" if "val_loss" in entry else ""
            print(f"Epoch {epoch:3d}/{epochs} | train loss {entry['train_loss']:.5f}{val_note}")
    return history


@torch.no_grad()
def score_rows(rows, model, transform, image_size, loss_name, blur_sigma, use_foreground_mask, device):
    """Return per-image anomaly maps (CPU, H x W) and reconstructions."""
    model.eval()
    anomaly_maps, reconstructions = [], []
    for _, row in rows.iterrows():
        image = Image.open(PROJECT_ROOT / row["image_path"]).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)
        reconstruction = model(input_tensor)
        anomaly_map = reconstruction_error_map(
            input_tensor, reconstruction, loss_name, blur_sigma
        )[0].cpu()
        if use_foreground_mask:
            resized_image = image.resize((image_size, image_size))
            foreground = torch.from_numpy(
                compute_foreground_mask(resized_image, image_size)
            ).bool()
            if foreground.any():
                anomaly_map = torch.where(foreground, anomaly_map, anomaly_map.min())
        anomaly_maps.append(anomaly_map)
        reconstructions.append(reconstruction[0].cpu())
    return anomaly_maps, reconstructions


def scores_from_maps(anomaly_maps, reduction: str, topk_fraction: float) -> np.ndarray:
    return np.array(
        [image_score(m, reduction, topk_fraction).item() for m in anomaly_maps]
    )


def plot_loss_curve(history, output_path: Path, title: str) -> None:
    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, [h["train_loss"] for h in history], label="train (fit subset)")
    if "val_loss" in history[0]:
        ax.plot(epochs, [h["val_loss"] for h in history], label="held-out normal (calibration)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reconstruction loss")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_reconstruction_triptych(
    image: Image.Image,
    reconstruction: torch.Tensor,
    anomaly_map: torch.Tensor,
    gt_mask,
    output_path: Path,
    title: str,
    vmin: float,
    vmax: float,
) -> None:
    panels = 4 if gt_mask is not None else 3
    fig, axes = plt.subplots(1, panels, figsize=(4 * panels, 4))
    axes[0].imshow(np.array(image))
    axes[0].set_title("Input")
    axes[1].imshow(reconstruction.permute(1, 2, 0).clamp(0, 1).numpy())
    axes[1].set_title("Reconstruction")
    axes[2].imshow(anomaly_map.numpy(), cmap="jet", vmin=vmin, vmax=vmax)
    axes[2].set_title("Reconstruction error")
    if gt_mask is not None:
        axes[3].imshow(gt_mask, cmap="gray")
        axes[3].set_title("Ground-truth mask")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    category = config.get("category", "screw")
    data_cfg = config["data"]
    anomaly_cfg = config.get("anomaly_detection", {})
    ae_cfg = config.get("autoencoder", {})
    output_cfg = config["output"]

    loss_name = args.loss or ae_cfg.get("loss", "l2")
    default_latent_dim = ae_cfg.get("latent_dim", 128)
    latent_dim = args.latent_dim or default_latent_dim
    epochs = args.epochs or ae_cfg.get("epochs", 200)
    score_reduction = args.score_reduction or ae_cfg.get("score_reduction", "mean")
    topk_fraction = float(ae_cfg.get("topk_fraction", 0.01))
    blur_sigma = float(ae_cfg.get("error_blur_sigma", 4.0))
    image_size = data_cfg["image_size"]
    seed = anomaly_cfg.get("seed", data_cfg["seed"])

    manifest = load_manifest(PROJECT_ROOT / data_cfg["manifest_path"])
    train_rows = manifest[(manifest["split"] == "train") & (manifest["label"] == 0)].reset_index(drop=True)
    test_rows = manifest[manifest["split"] == "test"].reset_index(drop=True)

    # Identical split/seed to PatchCore's so both fit on the same normals.
    calibration_cfg = anomaly_cfg.get("calibration", {})
    calibration_enabled = calibration_cfg.get("enabled", DEFAULT_CALIBRATION_ENABLED)
    holdout_ratio = calibration_cfg.get("holdout_ratio", DEFAULT_CALIBRATION_HOLDOUT_RATIO)
    calibration_percentile = calibration_cfg.get("percentile", DEFAULT_CALIBRATION_PERCENTILE)
    calibration_seed = calibration_cfg.get("seed", seed)
    if calibration_enabled:
        fit_rows, calibration_rows = split_calibration_rows(train_rows, holdout_ratio, calibration_seed)
        print(f"Calibration split: {len(fit_rows)} fitting / {len(calibration_rows)} held-out normal images")
    else:
        fit_rows, calibration_rows = train_rows, None

    transform = get_autoencoder_transforms(image_size)
    torch.manual_seed(seed)
    train_loader = DataLoader(
        preload(fit_rows, transform),
        batch_size=ae_cfg.get("batch_size", 16),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = None
    if calibration_enabled:
        val_loader = DataLoader(
            preload(calibration_rows, transform),
            batch_size=ae_cfg.get("batch_size", 16),
            shuffle=False,
        )

    device = get_device()
    model = ConvAutoencoder(
        image_size=image_size,
        latent_dim=latent_dim,
        base_channels=ae_cfg.get("base_channels", 32),
    ).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(
        f"Using device: {device} | loss: {loss_name} | latent dim: {latent_dim} | "
        f"epochs: {epochs} | params: {num_params / 1e6:.2f}M"
    )

    train_start = time.perf_counter()
    history = train_autoencoder(
        model,
        train_loader,
        val_loader,
        loss_name,
        epochs,
        float(ae_cfg.get("learning_rate", 2e-4)),
        float(ae_cfg.get("weight_decay", 0.0)),
        device,
    )
    synchronize(device)
    train_seconds = time.perf_counter() - train_start
    print(f"Training took {train_seconds:.1f}s")

    use_foreground_mask = anomaly_cfg.get("use_foreground_mask", True)
    score_args = (transform, image_size, loss_name, blur_sigma, use_foreground_mask, device)

    image_threshold = pixel_threshold = None
    if calibration_enabled:
        calibration_maps, _ = score_rows(calibration_rows, model, *score_args)
        calibration_scores = scores_from_maps(calibration_maps, score_reduction, topk_fraction)
        image_threshold = percentile_threshold(calibration_scores, calibration_percentile)
        pixel_threshold = percentile_threshold(
            np.concatenate([m.numpy().ravel() for m in calibration_maps]), calibration_percentile
        )
        print(f"Calibrated thresholds -- image: {image_threshold:.6f}, pixel: {pixel_threshold:.6f}")
        print("Freezing model + thresholds; evaluating once on the full test set...")

    synchronize(device)
    inference_start = time.perf_counter()
    anomaly_maps, reconstructions = score_rows(test_rows, model, *score_args)
    synchronize(device)
    inference_ms_per_image = 1000 * (time.perf_counter() - inference_start) / len(test_rows)

    labels_arr = test_rows["label"].astype(int).to_numpy()
    scores_by_reduction = {
        reduction: scores_from_maps(anomaly_maps, reduction, topk_fraction)
        for reduction in SCORE_REDUCTIONS
    }
    scores_arr = scores_by_reduction[score_reduction]
    auroc = roc_auc_score(labels_arr, scores_arr)
    threshold = image_threshold if calibration_enabled else youden_threshold(labels_arr, scores_arr)
    predictions = (scores_arr >= threshold).astype(int)

    metrics = compute_classification_metrics(labels_arr, predictions)
    metrics["auroc"] = float(auroc)
    metrics["auroc_by_score_reduction"] = {
        reduction: float(roc_auc_score(labels_arr, s)) for reduction, s in scores_by_reduction.items()
    }
    metrics["threshold"] = float(threshold)
    metrics["threshold_method"] = "calibration_percentile" if calibration_enabled else "youden_test"
    metrics["calibration"] = {
        "enabled": calibration_enabled,
        "holdout_ratio": holdout_ratio,
        "percentile": calibration_percentile,
        "seed": calibration_seed,
        "num_fit_images": len(fit_rows),
        "num_calibration_images": len(calibration_rows) if calibration_enabled else 0,
    }
    metrics["per_defect_type"] = compute_per_defect_type_detection(
        test_rows["defect_type"].tolist(), labels_arr, scores_arr, predictions
    )
    metrics.update(
        {
            "method": "autoencoder",
            "loss": loss_name,
            "latent_dim": latent_dim,
            "epochs": epochs,
            "score_reduction": score_reduction,
            "topk_fraction": topk_fraction,
            "error_blur_sigma": blur_sigma,
            "num_parameters": num_params,
            "final_train_loss": history[-1]["train_loss"],
            "final_heldout_normal_loss": history[-1].get("val_loss"),
            "train_seconds": train_seconds,
            "inference_ms_per_image": inference_ms_per_image,
            "device": device,
            "score_min": float(scores_arr.min()),
            "score_max": float(scores_arr.max()),
        }
    )

    gt_masks = [
        load_gt_mask(
            PROJECT_ROOT / row["mask_path"] if isinstance(row["mask_path"], str) and row["mask_path"] else None,
            image_size,
        )
        for _, row in test_rows.iterrows()
    ]
    pixel_metrics = compute_pixel_level_metrics(
        [m.numpy() for m in anomaly_maps],
        gt_masks,
        pixel_threshold=pixel_threshold if calibration_enabled else None,
    )
    display_maps = anomaly_maps
    keep_primary_region = anomaly_cfg.get("keep_primary_region", False)
    peak_fraction = anomaly_cfg.get("primary_region_peak_fraction", 0.0)
    metrics["keep_primary_region"] = keep_primary_region
    metrics["primary_region_peak_fraction"] = peak_fraction
    if keep_primary_region:
        display_maps = [
            keep_primary_anomaly_region(m, pixel_metrics["pixel_threshold"], peak_fraction=peak_fraction)
            for m in anomaly_maps
        ]
        filtered = compute_pixel_level_metrics(
            [m.numpy() for m in display_maps], gt_masks, pixel_threshold=pixel_metrics["pixel_threshold"]
        )
        pixel_metrics["raw_mean_iou"] = pixel_metrics["mean_iou"]
        pixel_metrics["raw_mean_dice"] = pixel_metrics["mean_dice"]
        pixel_metrics["postprocessed_pixel_auroc"] = filtered["pixel_auroc"]
        pixel_metrics["mean_iou"] = filtered["mean_iou"]
        pixel_metrics["mean_dice"] = filtered["mean_dice"]
    metrics.update(pixel_metrics)

    print(f"Image-level ROC-AUC ({score_reduction}): {auroc:.4f}")
    print(f"Image ROC-AUC by reduction: {metrics['auroc_by_score_reduction']}")
    print(f"Pixel-level ROC-AUC: {pixel_metrics['pixel_auroc']:.4f}")
    print(f"Mean IoU: {pixel_metrics['mean_iou']:.4f} | Mean Dice: {pixel_metrics['mean_dice']:.4f}")
    print(json.dumps(metrics, indent=2))

    checkpoint_dir = PROJECT_ROOT / output_cfg["checkpoint_dir"]
    metrics_dir = PROJECT_ROOT / output_cfg["metrics_dir"]
    heatmaps_dir = PROJECT_ROOT / output_cfg.get("heatmaps_dir", "outputs/heatmaps")
    figures_dir = PROJECT_ROOT / output_cfg.get("figures_dir", "outputs/figures")
    for directory in (checkpoint_dir, metrics_dir, heatmaps_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    run_name = build_run_name(category, loss_name, latent_dim, default_latent_dim)
    metrics_path = metrics_dir / f"{run_name}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    (metrics_dir / f"{run_name}_history.json").write_text(json.dumps(history, indent=2))

    title = f"Autoencoder-{loss_name.upper()} ({category})"
    plot_loss_curve(history, figures_dir / f"{run_name}_loss_curve.png", f"{title} — reconstruction loss")
    plot_confusion_matrix(
        labels_arr, predictions,
        output_path=figures_dir / f"{run_name}_confusion_matrix.png",
        title=f"{title} — confusion matrix",
    )
    plot_score_distribution(
        scores_arr, labels_arr, threshold=threshold,
        output_path=figures_dir / f"{run_name}_score_distribution.png",
        title=f"{title} — anomaly score distribution",
    )
    plot_metric_vs_threshold(
        labels_arr, scores_arr, chosen_threshold=threshold,
        output_path=figures_dir / f"{run_name}_metric_vs_threshold.png",
        title=f"{title} — metrics vs threshold",
    )
    print(f"Saved metrics to {metrics_path} and figures to {figures_dir}")

    if args.metrics_only:
        return

    torch.save(
        {
            "state_dict": model.state_dict(),
            "image_size": image_size,
            "latent_dim": latent_dim,
            "base_channels": ae_cfg.get("base_channels", 32),
            "loss": loss_name,
            "image_threshold": threshold,
            "pixel_threshold": pixel_metrics["pixel_threshold"],
        },
        checkpoint_dir / f"{run_name}.pt",
    )

    vmin = float(min(m.min().item() for m in display_maps))
    vmax = float(max(m.max().item() for m in display_maps))
    seen = set()
    for idx, row in test_rows.iterrows():
        defect_type = row["defect_type"]
        if defect_type in seen:
            continue
        seen.add(defect_type)
        image = Image.open(PROJECT_ROOT / row["image_path"]).convert("RGB").resize((image_size, image_size))
        score = float(scores_arr[idx])
        gt_mask = gt_masks[idx] if defect_type != "good" else None
        save_anomaly_heatmap(
            image, display_maps[idx], heatmaps_dir / f"{run_name}_{defect_type}_example.png",
            score=score, vmin=vmin, vmax=vmax, gt_mask=gt_mask,
            threshold=pixel_metrics["pixel_threshold"],
        )
        save_reconstruction_triptych(
            image, reconstructions[idx], anomaly_maps[idx], gt_mask,
            heatmaps_dir / f"{run_name}_{defect_type}_reconstruction.png",
            title=f"{title} — {defect_type} (score={score:.4g}, "
            f"{'Defective' if score >= threshold else 'Good'})",
            vmin=vmin, vmax=vmax,
        )
        print(f"[{defect_type}] score {score:.5f} -> {'Defective' if score >= threshold else 'Good'}")
    print(f"Saved checkpoint to {checkpoint_dir / f'{run_name}.pt'} and examples to {heatmaps_dir}")


if __name__ == "__main__":
    main()
