"""Benchmark defect-type classifiers with matched splits and D4 augmentation.

Runs a no-augmentation ResNet18 baseline, then compares D4-augmented ResNet18,
ConvNeXt-Tiny, and EfficientNet-B0 models across the nine project categories.
Validation probabilities are also combined with an equal-weight soft vote.
"""

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader

from src.data.dataset import ManifestImageDataset
from src.models.baseline_classifier import build_baseline_model
from src.models.train_defect_classifier import (
    PROJECT_ROOT,
    build_defect_manifest,
    train_with_early_stopping,
)
from src.preprocessing.transform import (
    get_defect_classifier_train_transforms,
    get_val_transforms,
)

CATEGORIES = (
    "screw",
    "bottle",
    "hazelnut",
    "carpet",
    "leather",
    "wood",
    "grid",
    "tile",
    "transistor",
)
ARCHITECTURES = ("resnet18", "convnext_tiny", "efficientnet_b0")
RUNS = (
    ("resnet18", "none"),
    ("resnet18", "d4"),
    ("convnext_tiny", "d4"),
    ("efficientnet_b0", "d4"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--categories", nargs="+", choices=CATEGORIES, default=CATEGORIES)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/metrics"))
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser.parse_args()


def predict_probabilities(model, loader, device):
    model.eval()
    all_probabilities = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            all_probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            all_labels.extend(labels.tolist())
    return np.concatenate(all_probabilities), np.asarray(all_labels)


def classification_scores(y_true, probabilities):
    predictions = probabilities.argmax(axis=1)
    return {
        "val_accuracy": float(accuracy_score(y_true, predictions)),
        "val_macro_f1": float(
            f1_score(y_true, predictions, average="macro", zero_division=0)
        ),
    }


def sample_std(values):
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def metric_record(category, architecture, augmentation, y_true, probabilities, history, best_epoch):
    return {
        "category": category,
        "architecture": architecture,
        "augmentation": augmentation,
        **classification_scores(y_true, probabilities),
        "final_train_loss": float(history[-1]["train_loss"]),
        "train_loss_at_best_epoch": float(history[best_epoch - 1]["train_loss"]),
        "best_epoch": int(best_epoch),
    }


def run_category(category, epochs_override, device):
    config_path = PROJECT_ROOT / "config" / f"{category}_config.yaml"
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    data_cfg = config["data"]
    train_cfg = config["train"]
    manifest, defect_types = build_defect_manifest(
        PROJECT_ROOT / data_cfg["manifest_path"]
    )
    train_rows, val_rows = train_test_split(
        manifest,
        test_size=data_cfg["val_split"],
        random_state=data_cfg["seed"],
        stratify=manifest["label"],
    )
    train_rows = train_rows.reset_index(drop=True)
    val_rows = val_rows.reset_index(drop=True)
    val_paths = val_rows["image_path"].tolist()
    print(
        f"\n{category}: {len(train_rows)} train / {len(val_rows)} validation; "
        f"per-class counts={manifest['label'].value_counts().sort_index().to_dict()}"
    )

    image_size = data_cfg["image_size"]
    val_dataset = ManifestImageDataset(
        val_rows, PROJECT_ROOT, transform=get_val_transforms(image_size)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
    )

    category_records = []
    probabilities_by_run = {}
    y_true_reference = None
    for architecture, augmentation in RUNS:
        seed = int(data_cfg["seed"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        train_transform = (
            get_defect_classifier_train_transforms(image_size)
            if augmentation == "d4"
            else get_val_transforms(image_size)
        )
        train_dataset = ManifestImageDataset(
            train_rows, PROJECT_ROOT, transform=train_transform
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_cfg["batch_size"],
            shuffle=True,
            num_workers=train_cfg["num_workers"],
        )

        model_config = copy.deepcopy(config["model"])
        model_config["architecture"] = architecture
        model = build_baseline_model(
            architecture=architecture,
            num_classes=len(defect_types),
            pretrained=model_config["pretrained"],
        ).to(device)
        epochs = epochs_override if epochs_override is not None else train_cfg["epochs"]
        criterion = nn.CrossEntropyLoss()
        learning_rate = train_cfg["learning_rate"]
        print(f"  Training {architecture} ({augmentation}), up to {epochs} epochs")
        _, _, training_summary = train_with_early_stopping(
            model,
            train_loader,
            val_loader,
            train_cfg,
            learning_rate,
            epochs,
            device,
        )
        probabilities, y_true = predict_probabilities(model, val_loader, device)
        if y_true_reference is None:
            y_true_reference = y_true
        elif not np.array_equal(y_true_reference, y_true):
            raise RuntimeError("Validation labels changed between benchmark runs")

        history = training_summary["history"]
        best_epoch = training_summary["best_epoch"]
        record = metric_record(
            category,
            architecture,
            augmentation,
            y_true,
            probabilities,
            history,
            best_epoch,
        )
        record.update(
            {
                "train_images": len(train_rows),
                "val_images": len(val_rows),
                "defect_types": len(defect_types),
                "seed": seed,
            }
        )
        category_records.append(record)
        probabilities_by_run[(architecture, augmentation)] = probabilities
        print(
            f"    val_accuracy={record['val_accuracy']:.4f}, "
            f"macro_f1={record['val_macro_f1']:.4f}, "
            f"final_train_loss={record['final_train_loss']:.4f}"
        )

    augmented_probabilities = np.mean(
        [
            probabilities_by_run[(architecture, "d4")]
            for architecture in ARCHITECTURES
        ],
        axis=0,
    )
    ensemble_record = {
        "category": category,
        "architecture": "soft_vote_3_models",
        "augmentation": "d4",
        **classification_scores(y_true_reference, augmented_probabilities),
        "train_images": len(train_rows),
        "val_images": len(val_rows),
        "defect_types": len(defect_types),
        "seed": int(data_cfg["seed"]),
    }
    print(
        f"  Soft vote: val_accuracy={ensemble_record['val_accuracy']:.4f}, "
        f"macro_f1={ensemble_record['val_macro_f1']:.4f}"
    )
    return category_records, ensemble_record


def main() -> None:
    args = parse_args()
    if args.epochs is not None and args.epochs < 1:
        raise ValueError("--epochs must be positive")
    if args.torch_threads < 1:
        raise ValueError("--torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}; CPU threads: {args.torch_threads}")
    print("Validation metrics are exploratory: all rows come from MVTec's labeled test split.")

    all_records = []
    ensemble_records = []
    for category in args.categories:
        category_records, ensemble_record = run_category(category, args.epochs, device)
        all_records.extend(category_records)
        ensemble_records.append(ensemble_record)

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = pd.DataFrame(all_records)
    ensembles = pd.DataFrame(ensemble_records)
    group_columns = ["architecture", "augmentation"]
    summary = (
        runs.groupby(group_columns, as_index=False)
        .agg(
            categories=("category", "nunique"),
            mean_val_accuracy=("val_accuracy", "mean"),
            std_val_accuracy=("val_accuracy", "std"),
            mean_val_macro_f1=("val_macro_f1", "mean"),
            std_val_macro_f1=("val_macro_f1", "std"),
            mean_final_train_loss=("final_train_loss", "mean"),
        )
    )
    summary["std_val_accuracy"] = summary["std_val_accuracy"].fillna(0.0)
    summary["std_val_macro_f1"] = summary["std_val_macro_f1"].fillna(0.0)
    ensemble_summary = {
        "categories": int(ensembles["category"].nunique()),
        "mean_val_accuracy": float(ensembles["val_accuracy"].mean()),
        "std_val_accuracy": sample_std(ensembles["val_accuracy"].tolist()),
        "mean_val_macro_f1": float(ensembles["val_macro_f1"].mean()),
        "std_val_macro_f1": sample_std(ensembles["val_macro_f1"].tolist()),
    }

    runs.to_csv(output_dir / "defect_classifier_augmentation_runs.csv", index=False)
    summary.to_csv(output_dir / "defect_classifier_augmentation_architecture_summary.csv", index=False)
    ensembles.to_csv(output_dir / "defect_classifier_augmentation_ensemble.csv", index=False)
    results = {
        "device": str(device),
        "seed_policy": "category config seed reset before each model run",
        "split_policy": "one stratified train/validation split per category, reused for every run",
        "augmentation": "random horizontal/vertical flips and exact rotations in {0, 90, 180, 270} degrees; training only",
        "baseline": "ResNet18 with no random augmentation",
        "validation_caveat": "Validation images are drawn from MVTec's labeled test split; use independent data or nested CV for an unbiased final estimate.",
        "runs": all_records,
        "architecture_summary": summary.to_dict(orient="records"),
        "ensemble_by_category": ensemble_records,
        "ensemble_summary": ensemble_summary,
    }
    (output_dir / "defect_classifier_augmentation_benchmark.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print("\nArchitecture summary (category means):")
    print(summary.to_string(index=False))
    print("\nSoft-voting ensemble summary:")
    print(json.dumps(ensemble_summary, indent=2))
    print(f"\nSaved benchmark files under {output_dir}")


if __name__ == "__main__":
    main()
