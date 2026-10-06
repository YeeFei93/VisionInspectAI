"""Benchmark defect-type classifiers with matched splits and deterministic augmentation.

Runs a no-augmentation ResNet18 baseline, then compares ResNet18, ConvNeXt-Tiny,
EfficientNet-B0, and DenseNet121 trained on fixed flip/rotation variants of each
training image (see get_defect_augmentation_variants) across the nine categories.
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.data.dataset import ManifestImageDataset
from src.models.baseline_classifier import build_baseline_model
from src.models.train_defect_classifier import (
    PROJECT_ROOT,
    build_defect_manifest,
    train_with_early_stopping,
)
from src.preprocessing.transform import (
    get_defect_augmentation_variants,
    get_defect_variant_transform,
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
OUTPUT_PREFIX = "defect_classifier_variant_augmentation"
SUPPORTED_ARCHITECTURES = (
    "resnet18",
    "convnext_tiny",
    "efficientnet_b0",
    "densenet121",
)


def expand_training_rows(rows, category):
    """One row per fixed augmentation variant of each training image."""
    expanded = []
    for row in rows.to_dict(orient="records"):
        for variant in get_defect_augmentation_variants(category, row["defect_type"]):
            expanded.append({**row, "augmentation_variant": variant})
    return pd.DataFrame(expanded)


class VariantDataset(Dataset):
    """Applies each expanded row's fixed variant to its (untransformed) image."""

    def __init__(self, expanded_rows, image_size):
        self.base = ManifestImageDataset(expanded_rows, PROJECT_ROOT)
        self.variants = expanded_rows["augmentation_variant"].tolist()
        self.transforms = {
            variant: get_defect_variant_transform(variant, image_size)
            for variant in set(self.variants)
        }

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image, label = self.base[index]
        return self.transforms[self.variants[index]](image), label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--categories", nargs="+", choices=CATEGORIES, default=CATEGORIES)
    parser.add_argument(
        "--architectures",
        nargs="+",
        choices=SUPPORTED_ARCHITECTURES,
        default=SUPPORTED_ARCHITECTURES,
    )
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--save-deployment-checkpoints", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/metrics"))
    parser.add_argument("--output-prefix", default=OUTPUT_PREFIX)
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser.parse_args()


def format_elapsed(seconds):
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


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


def save_deployment_artifacts(model, checkpoint_path: Path, metrics_path: Path, metadata):
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    metrics_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


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


def run_category(
    category,
    architectures,
    include_baseline,
    save_deployment_checkpoints,
    epochs_override,
    device,
    category_index,
    category_count,
    benchmark_start,
):
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
    expanded_train_rows = expand_training_rows(train_rows, category)
    print(
        f"\n{category}: {len(train_rows)} train ({len(expanded_train_rows)} with variants) / "
        f"{len(val_rows)} validation; "
        f"per-class counts={manifest['label'].value_counts().sort_index().to_dict()}",
        flush=True,
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

    runs = ([ ("resnet18", "none") ] if include_baseline else [])
    runs.extend((architecture, "variants") for architecture in architectures)
    category_records = []
    y_true_reference = None
    for model_index, (architecture, augmentation) in enumerate(runs, start=1):
        model_start = time.perf_counter()
        print(
            f"PROGRESS category={category_index}/{category_count} {category} "
            f"model={model_index}/{len(runs)} {architecture} status=START "
            f"elapsed={format_elapsed(model_start - benchmark_start)}",
            flush=True,
        )
        seed = int(data_cfg["seed"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        train_dataset = (
            VariantDataset(expanded_train_rows, image_size)
            if augmentation == "variants"
            else ManifestImageDataset(
                train_rows, PROJECT_ROOT, transform=get_val_transforms(image_size)
            )
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_cfg["batch_size"],
            shuffle=True,
            num_workers=train_cfg["num_workers"],
        )

        model = build_baseline_model(
            architecture=architecture,
            num_classes=len(defect_types),
            pretrained=config["model"]["pretrained"],
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
                "train_images_with_variants": len(train_dataset),
                "val_images": len(val_rows),
                "defect_types": len(defect_types),
                "seed": seed,
            }
        )
        category_records.append(record)
        if save_deployment_checkpoints and augmentation == "variants":
            output_cfg = config["output"]
            run_name = f"defect_classifier_{architecture}_{category}"
            checkpoint_dir = PROJECT_ROOT / output_cfg["checkpoint_dir"]
            metrics_dir = PROJECT_ROOT / output_cfg["metrics_dir"]
            deployment_metadata = {
                **record,
                "defect_types": defect_types,
                "pretrained": bool(config["model"]["pretrained"]),
                "crop_mode": "full_image",
                "validation_crop_source": None,
                "crop_padding_ratio": 0.25,
                "min_crop_fraction": 0.25,
                "image_size": image_size,
                "early_stopping_patience": int(
                    train_cfg.get("early_stopping_patience", 3)
                ),
                "checkpoint": f"{run_name}.pt",
            }
            save_deployment_artifacts(
                model,
                checkpoint_dir / f"{run_name}.pt",
                metrics_dir / f"{run_name}_metrics.json",
                deployment_metadata,
            )
        print(
            f"    val_accuracy={record['val_accuracy']:.4f}, "
            f"macro_f1={record['val_macro_f1']:.4f}, "
            f"final_train_loss={record['final_train_loss']:.4f}",
            flush=True,
        )
        print(
            f"PROGRESS category={category_index}/{category_count} {category} "
            f"model={model_index}/{len(runs)} {architecture} status=DONE "
            f"model_elapsed={format_elapsed(time.perf_counter() - model_start)} "
            f"total_elapsed={format_elapsed(time.perf_counter() - benchmark_start)}",
            flush=True,
        )

    return category_records


def main() -> None:
    args = parse_args()
    if args.epochs is not None and args.epochs < 1:
        raise ValueError("--epochs must be positive")
    if args.torch_threads < 1:
        raise ValueError("--torch-threads must be positive")
    if not args.architectures and args.skip_baseline:
        raise ValueError("Select at least one architecture or include the baseline")
    torch.set_num_threads(args.torch_threads)
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}; CPU threads: {args.torch_threads}")
    print("Validation metrics are exploratory: all rows come from MVTec's labeled test split.", flush=True)
    benchmark_start = time.perf_counter()

    all_records = []
    for category_index, category in enumerate(args.categories, start=1):
        all_records.extend(
            run_category(
                category,
                args.architectures,
                not args.skip_baseline,
                args.save_deployment_checkpoints,
                args.epochs,
                device,
                category_index,
                len(args.categories),
                benchmark_start,
            )
        )

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = pd.DataFrame(all_records)
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

    runs.to_csv(output_dir / f"{args.output_prefix}_runs.csv", index=False)
    summary.to_csv(output_dir / f"{args.output_prefix}_architecture_summary.csv", index=False)
    results = {
        "device": str(device),
        "seed_policy": "category config seed reset before each model run",
        "split_policy": "one stratified train/validation split per category, reused for every run",
        "augmentation": "deterministic per-image variants (flips and exact 90/180/270 rotations; wood flips only; transistor restricted); training only",
        "architectures": args.architectures,
        "baseline": "ResNet18 with no augmentation" if not args.skip_baseline else None,
        "deployment_checkpoints_saved": args.save_deployment_checkpoints,
        "validation_caveat": "Validation images are drawn from MVTec's labeled test split; use independent data or nested CV for an unbiased final estimate.",
        "runs": all_records,
        "architecture_summary": summary.to_dict(orient="records"),
    }
    (output_dir / f"{args.output_prefix}_benchmark.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print("\nArchitecture summary (category means):")
    print(summary.to_string(index=False))
    print(
        f"\nSaved benchmark files under {output_dir}; "
        f"total elapsed={format_elapsed(time.perf_counter() - benchmark_start)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
