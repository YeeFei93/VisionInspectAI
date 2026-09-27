"""Train a per-category "what kind of defect is this?" classifier.

Unlike the baseline good-vs-defective classifier, this only looks at the
*defective* rows of one category's manifest (test/ split, label == 1) and
predicts which defect_type it is (e.g. leather: color / cut / fold / glue /
poke). Used by the Streamlit demo to show a defect type alongside the
Normal/Defective verdict and severity.

Usage:
    python -m src.models.train_defect_classifier --config config/leather_config.yaml

    Nested holdout + cross-validation workflow for tiny per-class counts
    (e.g. transistor: 4 defect types x 10 images):
        # 1-2. Reserve 2/class for final test, 4-fold CV on the remaining 32 to pick settings
        python -m src.models.train_defect_classifier --config config/transistor_config.yaml \\
            --freeze-backbone --holdout-per-class 2 --cross-validation --folds 4
        # 3-4. Retrain on all 32 dev images with the chosen settings, evaluate once on the 8 held out
        python -m src.models.train_defect_classifier --config config/transistor_config.yaml \\
            --freeze-backbone --holdout-per-class 2
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import nn
from torch.utils.data import DataLoader

from src.data.dataset import ManifestImageDataset, load_manifest
from src.evaluation.metrics import compute_per_class_report, plot_training_curves
from src.models.anomaly_detector import PatchCoreAnomalyDetector, scoring_artifact_suffix
from src.models.baseline_classifier import build_baseline_model, freeze_backbone
from src.models.defect_regions import keep_primary_anomaly_region
from src.preprocessing.segmentation import compute_foreground_mask
from src.preprocessing.transform import get_train_transforms, get_val_transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("config/screw_config.yaml")
    )
    parser.add_argument(
        "--defect-focused-crops",
        action="store_true",
        help="Train and validate on ground-truth-mask crops instead of full images.",
    )
    parser.add_argument("--crop-padding-ratio", type=float, default=0.25)
    parser.add_argument("--min-crop-fraction", type=float, default=0.25)
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze all layers except the classification head (overfitting control for tiny datasets).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override the config's train.learning_rate (a frozen head usually needs a higher LR than fine-tuning).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the config's train.epochs (e.g. to keep training past the default when val loss is still improving).",
    )
    parser.add_argument(
        "--crop-source",
        choices=("ground-truth", "patchcore"),
        default="ground-truth",
        help="Localization source for focused crops.",
    )
    parser.add_argument(
        "--holdout-per-class",
        type=int,
        default=0,
        help=(
            "Reserve this many images per defect_type as a final held-out test set, excluded "
            "from both --cross-validation and training (0 disables)."
        ),
    )
    parser.add_argument(
        "--cross-validation",
        action="store_true",
        help=(
            "Run stratified K-fold CV on the development set (post-holdout) for model "
            "selection. Does not save a checkpoint or touch the holdout set."
        ),
    )
    parser.add_argument("--folds", type=int, default=4, help="Number of cross-validation folds.")
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    with config_path.open() as f:
        return yaml.safe_load(f)


def build_defect_manifest(manifest_path: Path):
    """Defective (label == 1) rows of the labeled test/ split, with the
    good/defective "label" column replaced by a defect_type index."""
    manifest = load_manifest(manifest_path)
    defective = manifest[(manifest["split"] == "test") & (manifest["label"] == 1)].copy()
    defect_types = sorted(defective["defect_type"].unique())
    defective["label"] = defective["defect_type"].map(defect_types.index)
    return defective.reset_index(drop=True), defect_types


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
    dataset_size = len(loader.dataset)
    return running_loss / dataset_size, correct / dataset_size


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().tolist())
    dataset_size = len(loader.dataset)
    accuracy = sum(p == t for p, t in zip(all_preds, all_labels)) / dataset_size
    return running_loss / dataset_size, accuracy, all_labels, all_preds


def reserve_holdout_test_set(manifest, per_class: int, seed: int):
    """Split off `per_class` images per defect_type (stratified, seeded) into an
    untouched final test set, leaving the rest as the development set used for
    cross-validation and/or the final training run."""
    rng = np.random.RandomState(seed)
    holdout_frames = []
    for label in sorted(manifest["label"].unique()):
        class_rows = manifest[manifest["label"] == label]
        if len(class_rows) <= per_class:
            raise ValueError(
                f"--holdout-per-class={per_class} leaves no development data for label "
                f"{label} (only {len(class_rows)} images)"
            )
        holdout_frames.append(class_rows.sample(n=per_class, random_state=rng))
    holdout = pd.concat(holdout_frames)
    dev = manifest.drop(index=holdout.index)
    return dev.reset_index(drop=True), holdout.reset_index(drop=True)


def build_loaders(train_subset, val_subset, data_cfg, train_cfg, crop_kwargs):
    image_size = data_cfg["image_size"]
    train_dataset = ManifestImageDataset(
        train_subset, PROJECT_ROOT, transform=get_train_transforms(image_size), **crop_kwargs
    )
    val_dataset = ManifestImageDataset(
        val_subset, PROJECT_ROOT, transform=get_val_transforms(image_size), **crop_kwargs
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
    )
    return train_loader, val_loader


def train_with_early_stopping(model, train_loader, val_loader, train_cfg, learning_rate, epochs, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=learning_rate)
    patience = int(train_cfg.get("early_stopping_patience", 3))
    min_delta = float(train_cfg.get("early_stopping_min_delta", 0.0))

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy, _, _ = evaluate(model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "learning_rate": current_lr,
            }
        )
        print(
            f"Epoch {epoch}/{epochs} - "
            f"accuracy: {train_accuracy:.4f} - loss: {train_loss:.4f} - "
            f"val_accuracy: {val_accuracy:.4f} - val_loss: {val_loss:.4f} - "
            f"learning_rate: {current_lr:.4g}"
        )

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if patience > 0 and epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
                break

    model.load_state_dict(best_state)
    _val_loss, _val_accuracy, y_true, y_pred = evaluate(model, val_loader, criterion, device)
    training_summary = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "epochs_trained": len(history),
        "early_stopping_patience": patience,
        "history": history,
    }
    return y_true, y_pred, training_summary


def run_cross_validation(dev_manifest, defect_types, config, args, device, folds, crop_kwargs):
    """Stratified K-fold CV on the development set, for choosing training settings
    (e.g. freeze-backbone, learning rate, epochs) without ever touching the holdout
    test set. Reports per-fold and aggregate (mean/std) accuracy plus raw
    correct/total counts, since with tiny per-fold validation sizes a single
    misclassified image can swing accuracy by several percentage points."""
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["train"]

    if folds < 2:
        raise ValueError("--folds must be at least 2.")
    smallest_class = int(dev_manifest["label"].value_counts().min())
    if folds > smallest_class:
        raise ValueError(f"--folds cannot exceed the smallest class count ({smallest_class}).")

    epochs = args.epochs if args.epochs is not None else train_cfg["epochs"]
    learning_rate = args.learning_rate if args.learning_rate is not None else train_cfg["learning_rate"]

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=data_cfg["seed"])
    fold_results = []
    out_of_fold_true, out_of_fold_pred = [], []

    for fold, (train_indices, val_indices) in enumerate(
        splitter.split(dev_manifest, dev_manifest["label"]), start=1
    ):
        print(f"\n=== Fold {fold}/{folds} ===")
        torch.manual_seed(data_cfg["seed"] + fold)
        train_subset = dev_manifest.iloc[train_indices].reset_index(drop=True)
        val_subset = dev_manifest.iloc[val_indices].reset_index(drop=True)
        print(f"Train subset: {len(train_subset)} images | Val subset: {len(val_subset)} images")
        train_loader, val_loader = build_loaders(train_subset, val_subset, data_cfg, train_cfg, crop_kwargs)

        model = build_baseline_model(
            architecture=model_cfg["architecture"],
            num_classes=len(defect_types),
            pretrained=model_cfg["pretrained"],
        ).to(device)
        if args.freeze_backbone:
            model = freeze_backbone(model, model_cfg["architecture"])

        y_true, y_pred, training_summary = train_with_early_stopping(
            model, train_loader, val_loader, train_cfg, learning_rate, epochs, device
        )
        correct = int(sum(t == p for t, p in zip(y_true, y_pred)))
        fold_metrics = {
            "fold": fold,
            "accuracy": correct / len(y_true),
            "correct": correct,
            "total": len(y_true),
            "best_epoch": training_summary["best_epoch"],
            "best_val_loss": training_summary["best_val_loss"],
            "epochs_trained": training_summary["epochs_trained"],
        }
        fold_results.append(fold_metrics)
        out_of_fold_true.extend(y_true)
        out_of_fold_pred.extend(y_pred)
        print(json.dumps(fold_metrics, indent=2))

    accuracies = [fold["accuracy"] for fold in fold_results]
    out_of_fold_correct = int(sum(t == p for t, p in zip(out_of_fold_true, out_of_fold_pred)))
    return {
        "folds": folds,
        "seed": data_cfg["seed"],
        "development_set_size": len(dev_manifest),
        "freeze_backbone": args.freeze_backbone,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "aggregate": {
            "accuracy_mean": float(np.mean(accuracies)),
            "accuracy_std": float(np.std(accuracies, ddof=1)) if len(accuracies) > 1 else 0.0,
        },
        "out_of_fold_accuracy": out_of_fold_correct / len(out_of_fold_true),
        "out_of_fold_correct": out_of_fold_correct,
        "out_of_fold_total": len(out_of_fold_true),
        "out_of_fold_confusion_matrix": confusion_matrix(out_of_fold_true, out_of_fold_pred).tolist(),
        "fold_results": fold_results,
    }


def add_patchcore_focus_masks(manifest, config: dict):
    """Attach deployment-equivalent PatchCore masks before splitting/training."""
    data_cfg = config["data"]
    anomaly_cfg = config["anomaly_detection"]
    output_cfg = config["output"]

    np.random.seed(data_cfg["seed"])
    torch.manual_seed(data_cfg["seed"])
    category = config["category"]
    run_name = f"patchcore_{anomaly_cfg['backbone']}_{category}"
    checkpoint_path = PROJECT_ROOT / output_cfg["checkpoint_dir"] / f"{run_name}_memory_bank.pt"
    metrics_name = run_name + scoring_artifact_suffix(
        anomaly_cfg.get("num_neighbors", 1),
        anomaly_cfg.get("softmax_reweighting", False),
        anomaly_cfg.get("reweight_num_neighbors", 9),
    )
    metrics_path = PROJECT_ROOT / output_cfg["metrics_dir"] / f"{metrics_name}_metrics.json"
    if not checkpoint_path.exists() or not metrics_path.exists():
        raise FileNotFoundError(
            "PatchCore crop training requires the default detector checkpoint and metrics"
        )
    detector = PatchCoreAnomalyDetector(
        backbone=anomaly_cfg["backbone"],
        layers=tuple(anomaly_cfg["layers"]),
        device="cpu",
        num_neighbors=anomaly_cfg.get("num_neighbors", 1),
        softmax_reweighting=anomaly_cfg.get("softmax_reweighting", False),
        reweight_num_neighbors=anomaly_cfg.get("reweight_num_neighbors", 9),
    )
    detector.load(checkpoint_path)
    pixel_threshold = float(json.loads(metrics_path.read_text())["pixel_threshold"])
    image_size = data_cfg["image_size"]
    transform = get_val_transforms(image_size)
    use_foreground_mask = anomaly_cfg.get("use_foreground_mask", True)
    focus_masks = []
    for _, row in manifest.iterrows():
        image = Image.open(PROJECT_ROOT / row["image_path"]).convert("RGB")
        resized_image = image.resize((image_size, image_size))
        if use_foreground_mask:
            foreground_mask = compute_foreground_mask(resized_image, image_size)
            foreground_tensor = torch.from_numpy(foreground_mask).unsqueeze(0)
        else:
            foreground_mask = np.ones((image_size, image_size), dtype=bool)
            foreground_tensor = None
        result = detector.predict(
            transform(image).unsqueeze(0), foreground_masks=foreground_tensor
        )[0]
        anomaly_map = result.anomaly_map
        if anomaly_cfg.get("keep_primary_region", False):
            anomaly_map = keep_primary_anomaly_region(
                anomaly_map,
                pixel_threshold,
                peak_fraction=anomaly_cfg.get(
                    "primary_region_peak_fraction", 0.0
                ),
            )
        focus_masks.append(
            np.logical_and(anomaly_map.numpy() >= pixel_threshold, foreground_mask)
        )
    focused_manifest = manifest.copy()
    focused_manifest["focus_mask"] = focus_masks
    return focused_manifest


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    category = config.get("category", "screw")
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["train"]
    output_cfg = config["output"]

    np.random.seed(data_cfg["seed"])
    torch.manual_seed(data_cfg["seed"])

    manifest, defect_types = build_defect_manifest(PROJECT_ROOT / data_cfg["manifest_path"])
    if args.defect_focused_crops and args.crop_source == "patchcore":
        print("Generating deployment-equivalent PatchCore focus masks...")
        manifest = add_patchcore_focus_masks(manifest, config)
    print(f"Category: {category} | Defect types (label order): {defect_types}")
    print(f"Total defective images: {len(manifest)}")
    print(f"Per-class counts: {manifest['label'].value_counts().sort_index().to_dict()}")

    image_size = data_cfg["image_size"]
    crop_kwargs = {
        "focus_on_mask": args.defect_focused_crops,
        "crop_padding_ratio": args.crop_padding_ratio,
        "min_crop_fraction": args.min_crop_fraction,
    }
    epochs = args.epochs if args.epochs is not None else train_cfg["epochs"]
    learning_rate = args.learning_rate if args.learning_rate is not None else train_cfg["learning_rate"]

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    checkpoint_dir = PROJECT_ROOT / output_cfg["checkpoint_dir"]
    metrics_dir = PROJECT_ROOT / output_cfg["metrics_dir"]
    figures_dir = PROJECT_ROOT / output_cfg["figures_dir"]
    for directory in (checkpoint_dir, metrics_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    run_name = f"defect_classifier_{model_cfg['architecture']}_{category}"
    if args.defect_focused_crops:
        run_name += "_focused"
        if args.crop_source == "patchcore":
            run_name += "_patchcore"
    if args.freeze_backbone:
        run_name += "_frozen"

    if args.holdout_per_class > 0:
        dev_manifest, holdout_manifest = reserve_holdout_test_set(
            manifest, args.holdout_per_class, data_cfg["seed"]
        )
        print(
            f"Reserved {len(holdout_manifest)} holdout images ({args.holdout_per_class}/class); "
            f"development set: {len(dev_manifest)} images"
        )
        print(f"Development per-class counts: {dev_manifest['label'].value_counts().sort_index().to_dict()}")
        print(f"Holdout per-class counts: {holdout_manifest['label'].value_counts().sort_index().to_dict()}")
    else:
        dev_manifest, holdout_manifest = manifest, None

    if args.cross_validation:
        cv_metrics = run_cross_validation(
            dev_manifest, defect_types, config, args, device, args.folds, crop_kwargs
        )
        print("\nCross-validation aggregate metrics:")
        print(json.dumps(cv_metrics["aggregate"], indent=2))
        print(
            f"Out-of-fold accuracy: {cv_metrics['out_of_fold_correct']}/{cv_metrics['out_of_fold_total']} "
            f"({cv_metrics['out_of_fold_accuracy']:.1%})"
        )
        cv_metrics_path = metrics_dir / f"{run_name}_cv{args.folds}_metrics.json"
        cv_metrics_path.write_text(json.dumps(cv_metrics, indent=2))
        print(f"Saved cross-validation metrics to {cv_metrics_path}")
        return

    model = build_baseline_model(
        architecture=model_cfg["architecture"],
        num_classes=len(defect_types),
        pretrained=model_cfg["pretrained"],
    ).to(device)
    if args.freeze_backbone:
        model = freeze_backbone(model, model_cfg["architecture"])
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Backbone frozen; trainable parameters: {trainable}")

    if holdout_manifest is not None:
        # Final fit: train on every development image (settings should already be
        # chosen via --cross-validation, so there is no internal val split/early
        # stopping here), then evaluate exactly once on the untouched holdout set.
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=learning_rate)
        train_dataset = ManifestImageDataset(
            dev_manifest, PROJECT_ROOT, transform=get_train_transforms(image_size), **crop_kwargs
        )
        train_loader = DataLoader(
            train_dataset, batch_size=train_cfg["batch_size"], shuffle=True, num_workers=train_cfg["num_workers"]
        )
        history = []
        for epoch in range(1, epochs + 1):
            train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
            history.append({"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_accuracy})
            print(f"Epoch {epoch}/{epochs} - accuracy: {train_accuracy:.4f} - loss: {train_loss:.4f}")

        holdout_dataset = ManifestImageDataset(
            holdout_manifest, PROJECT_ROOT, transform=get_val_transforms(image_size), **crop_kwargs
        )
        holdout_loader = DataLoader(
            holdout_dataset, batch_size=train_cfg["batch_size"], shuffle=False, num_workers=train_cfg["num_workers"]
        )
        _loss, _accuracy, y_true, y_pred = evaluate(model, holdout_loader, criterion, device)
        correct = int(sum(t == p for t, p in zip(y_true, y_pred)))
        total = len(y_true)
        accuracy = correct / total
        cm = confusion_matrix(y_true, y_pred).tolist()
        per_class_report = compute_per_class_report(y_true, y_pred, defect_types)
        print(f"Holdout test accuracy: {correct}/{total} ({accuracy:.1%})")
        print(f"Confusion matrix (rows=true, cols=pred, order={defect_types}): {cm}")
        print("Per-class report:")
        print(json.dumps(per_class_report, indent=2))

        checkpoint_path = checkpoint_dir / f"{run_name}.pt"
        torch.save(model.state_dict(), checkpoint_path)

        metrics_path = metrics_dir / f"{run_name}_metrics.json"
        metrics_path.write_text(
            json.dumps(
                {
                    "defect_types": defect_types,
                    "holdout_per_class": args.holdout_per_class,
                    "development_set_size": len(dev_manifest),
                    "holdout_test_accuracy": accuracy,
                    "holdout_test_correct": correct,
                    "holdout_test_total": total,
                    "confusion_matrix": cm,
                    "classification_report": per_class_report,
                    "history": history,
                    "seed": data_cfg["seed"],
                    "freeze_backbone": args.freeze_backbone,
                    "learning_rate": learning_rate,
                    "epochs_trained": len(history),
                    "crop_mode": "defect_focused" if args.defect_focused_crops else "full_image",
                    "validation_crop_source": (
                        f"{args.crop_source}_mask" if args.defect_focused_crops else None
                    ),
                    "crop_source": args.crop_source if args.defect_focused_crops else None,
                    "crop_padding_ratio": args.crop_padding_ratio,
                    "min_crop_fraction": args.min_crop_fraction,
                },
                indent=2,
            )
        )
        print(f"Saved checkpoint to {checkpoint_path}")
        print(f"Saved metrics to {metrics_path}")
        return

    train_subset, val_subset = train_test_split(
        dev_manifest, test_size=data_cfg["val_split"], random_state=data_cfg["seed"], stratify=dev_manifest["label"]
    )
    print(f"Train subset: {len(train_subset)} images | Val subset: {len(val_subset)} images")
    train_loader, val_loader = build_loaders(train_subset, val_subset, data_cfg, train_cfg, crop_kwargs)

    y_true, y_pred, training_summary = train_with_early_stopping(
        model, train_loader, val_loader, train_cfg, learning_rate, epochs, device
    )

    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred).tolist()
    per_class_report = compute_per_class_report(y_true, y_pred, defect_types)
    print(f"Validation accuracy (best epoch restored): {accuracy:.4f}")
    print(f"Confusion matrix (rows=true, cols=pred, order={defect_types}): {cm}")
    print("Per-class report:")
    print(json.dumps(per_class_report, indent=2))

    checkpoint_path = checkpoint_dir / f"{run_name}.pt"
    torch.save(model.state_dict(), checkpoint_path)

    metrics_path = metrics_dir / f"{run_name}_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "defect_types": defect_types,
                "val_accuracy": accuracy,
                "confusion_matrix": cm,
                "classification_report": per_class_report,
                "history": training_summary["history"],
                "seed": data_cfg["seed"],
                "freeze_backbone": args.freeze_backbone,
                "best_epoch": training_summary["best_epoch"],
                "best_val_loss": training_summary["best_val_loss"],
                "epochs_trained": training_summary["epochs_trained"],
                "early_stopping_patience": training_summary["early_stopping_patience"],
                "crop_mode": "defect_focused" if args.defect_focused_crops else "full_image",
                "validation_crop_source": (
                    f"{args.crop_source}_mask" if args.defect_focused_crops else None
                ),
                "crop_source": args.crop_source if args.defect_focused_crops else None,
                "crop_padding_ratio": args.crop_padding_ratio,
                "min_crop_fraction": args.min_crop_fraction,
            },
            indent=2,
        )
    )

    figure_path = figures_dir / f"{run_name}_training_curves.png"
    plot_training_curves(training_summary["history"], output_path=figure_path, title_prefix="Defect-type classifier")

    print(f"Saved checkpoint to {checkpoint_path}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved training curves figure to {figure_path}")


if __name__ == "__main__":
    main()
