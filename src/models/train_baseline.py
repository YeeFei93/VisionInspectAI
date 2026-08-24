"""Train the first baseline supervised good-vs-defective classifier.

The MVTec-AD train/ folder only contains "good" images, so this baseline
instead splits the labeled test/ rows (good + every defect type) into a
train subset and a held-out validation subset. This is a supervised
baseline only — the proper approach for this dataset is unsupervised
anomaly detection trained on train/good (see src/models/anomaly_detector.py).

Usage:
    python -m src.models.train_baseline --config config/screw_config.yaml
    python -m src.models.train_baseline --config config/screw_config.yaml \
        --cross-validation --folds 5
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader

from src.data.dataset import ManifestImageDataset, load_manifest, make_train_val_split
from src.evaluation.metrics import compute_classification_metrics, plot_confusion_matrix, plot_training_curves
from src.models.baseline_classifier import build_baseline_model
from src.preprocessing.transform import get_train_transforms, get_val_transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("config/screw_config.yaml")
    )
    parser.add_argument(
        "--cross-validation",
        action="store_true",
        help="Run stratified K-fold evaluation instead of training a deployable checkpoint.",
    )
    parser.add_argument("--folds", type=int, default=5, help="Number of cross-validation folds.")
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    with config_path.open() as f:
        return yaml.safe_load(f)


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


def build_loaders(train_subset, val_subset, data_cfg, train_cfg):
    image_size = data_cfg["image_size"]
    train_dataset = ManifestImageDataset(
        train_subset, PROJECT_ROOT, transform=get_train_transforms(image_size)
    )
    val_dataset = ManifestImageDataset(
        val_subset, PROJECT_ROOT, transform=get_val_transforms(image_size)
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


def train_with_early_stopping(model, train_loader, val_loader, train_cfg, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])
    patience = int(train_cfg.get("early_stopping_patience", 3))
    min_delta = float(train_cfg.get("early_stopping_min_delta", 0.0))

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, train_cfg["epochs"] + 1):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy, _, _ = evaluate(model, val_loader, criterion, device)
        learning_rate = optimizer.param_groups[0]["lr"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "learning_rate": learning_rate,
            }
        )
        print(
            f"Epoch {epoch}/{train_cfg['epochs']} - "
            f"accuracy: {train_accuracy:.4f} - loss: {train_loss:.4f} - "
            f"val_accuracy: {val_accuracy:.4f} - val_loss: {val_loss:.4f} - "
            f"learning_rate: {learning_rate:.4g}"
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
    val_loss, _val_accuracy, y_true, y_pred = evaluate(model, val_loader, criterion, device)
    training_summary = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "epochs_trained": len(history),
        "early_stopping_patience": patience,
        "history": history,
    }
    return y_true, y_pred, val_loss, training_summary


def build_model(model_cfg, device):
    return build_baseline_model(
        architecture=model_cfg["architecture"],
        num_classes=model_cfg["num_classes"],
        pretrained=model_cfg["pretrained"],
    ).to(device)


def run_cross_validation(manifest, data_cfg, model_cfg, train_cfg, device, folds):
    test_rows = manifest[manifest["split"] == "test"].reset_index(drop=True)
    if folds < 2:
        raise ValueError("--folds must be at least 2.")
    smallest_class = int(test_rows["label"].value_counts().min())
    if folds > smallest_class:
        raise ValueError(f"--folds cannot exceed the smallest class count ({smallest_class}).")

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=data_cfg["seed"])
    fold_results = []
    out_of_fold_true, out_of_fold_pred = [], []
    metric_names = ("accuracy", "precision", "recall", "f1")

    for fold, (train_indices, val_indices) in enumerate(
        splitter.split(test_rows, test_rows["label"]), start=1
    ):
        print(f"\n=== Fold {fold}/{folds} ===")
        torch.manual_seed(data_cfg["seed"] + fold)
        train_subset = test_rows.iloc[train_indices].reset_index(drop=True)
        val_subset = test_rows.iloc[val_indices].reset_index(drop=True)
        print(f"Train subset: {len(train_subset)} images | Val subset: {len(val_subset)} images")
        train_loader, val_loader = build_loaders(train_subset, val_subset, data_cfg, train_cfg)
        model = build_model(model_cfg, device)
        y_true, y_pred, _, training_summary = train_with_early_stopping(
            model, train_loader, val_loader, train_cfg, device
        )
        metrics = compute_classification_metrics(y_true, y_pred)
        metrics.update({key: value for key, value in training_summary.items() if key != "history"})
        metrics["fold"] = fold
        fold_results.append(metrics)
        out_of_fold_true.extend(y_true)
        out_of_fold_pred.extend(y_pred)
        print(json.dumps(metrics, indent=2))

    aggregate = {
        name: {
            "mean": float(np.mean([fold[name] for fold in fold_results])),
            "std": float(np.std([fold[name] for fold in fold_results], ddof=1)),
        }
        for name in metric_names
    }
    return {
        "folds": folds,
        "seed": data_cfg["seed"],
        "aggregate": aggregate,
        "out_of_fold_metrics": compute_classification_metrics(out_of_fold_true, out_of_fold_pred),
        "fold_results": fold_results,
    }, out_of_fold_true, out_of_fold_pred


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    category = config.get("category", "screw")
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["train"]
    output_cfg = config["output"]

    torch.manual_seed(data_cfg["seed"])
    manifest = load_manifest(PROJECT_ROOT / data_cfg["manifest_path"])
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

    run_name = f"baseline_{model_cfg['architecture']}_{category}"

    if args.cross_validation:
        metrics, y_true, y_pred = run_cross_validation(
            manifest, data_cfg, model_cfg, train_cfg, device, args.folds
        )
        metrics_path = metrics_dir / f"{run_name}_cv{args.folds}_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2))
        figure_path = figures_dir / f"{run_name}_cv{args.folds}_confusion_matrix.png"
        plot_confusion_matrix(y_true, y_pred, output_path=figure_path)
        print("\nCross-validation aggregate metrics:")
        print(json.dumps(metrics["aggregate"], indent=2))
        print(f"Saved cross-validation metrics to {metrics_path}")
        print(f"Saved out-of-fold confusion matrix figure to {figure_path}")
        return

    train_subset, val_subset = make_train_val_split(
        manifest, val_split=data_cfg["val_split"], seed=data_cfg["seed"]
    )
    print(f"Train subset: {len(train_subset)} images | Val subset: {len(val_subset)} images")
    train_loader, val_loader = build_loaders(train_subset, val_subset, data_cfg, train_cfg)
    model = build_model(model_cfg, device)
    y_true, y_pred, _, training_summary = train_with_early_stopping(
        model, train_loader, val_loader, train_cfg, device
    )
    metrics = compute_classification_metrics(y_true, y_pred)
    metrics.update(training_summary)
    print("Validation metrics (best epoch restored):")
    print(json.dumps(metrics, indent=2))

    checkpoint_path = checkpoint_dir / f"{run_name}.pt"
    torch.save(model.state_dict(), checkpoint_path)

    metrics_path = metrics_dir / f"{run_name}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    figure_path = figures_dir / f"{run_name}_confusion_matrix.png"
    plot_confusion_matrix(y_true, y_pred, output_path=figure_path)

    loss_curve_path = figures_dir / f"{run_name}_loss_curve.png"
    plot_training_curves(training_summary["history"], output_path=loss_curve_path)

    print(f"Saved checkpoint to {checkpoint_path}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved confusion matrix figure to {figure_path}")
    print(f"Saved loss curve figure to {loss_curve_path}")


if __name__ == "__main__":
    main()
