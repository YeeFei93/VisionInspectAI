"""Train a small "which MVTec-AD category is this?" classifier.

This lets the Streamlit demo auto-detect the object type from an uploaded
image (screw / bottle / hazelnut / ...) instead of requiring the user to
pick a category manually, then routes to that category's PatchCore
anomaly detector automatically.

Combines every manifest listed in --categories (default: all configured in
app/streamlit_app.py's CATEGORY_CONFIGS), using every row (train + test,
regardless of good/defective) since object-type recognition doesn't care
about defect status, and reuses ManifestImageDataset by treating the
category index as the "label" column.

Usage:
    python -m src.models.train_category_classifier
    python -m src.models.train_category_classifier --categories screw bottle hazelnut
"""

import argparse
import copy
import json
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from src.data.dataset import ManifestImageDataset, load_manifest
from src.evaluation.metrics import compute_per_class_report, plot_training_curves
from src.models.baseline_classifier import build_baseline_model
from src.preprocessing.transform import get_train_transforms, get_val_transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "checkpoints" / "category_classifier_resnet18.pt"
METRICS_PATH = PROJECT_ROOT / "outputs" / "metrics" / "category_classifier_metrics.json"
FIGURE_PATH = PROJECT_ROOT / "outputs" / "figures" / "category_classifier_resnet18_training_curves.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["screw", "bottle", "hazelnut", "carpet", "leather", "grid", "tile", "wood"],
        help="Categories to include (must each have data/manifests/<category>.csv).",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=3,
        help="Stop after this many epochs without val-loss improvement (0 disables).",
    )
    return parser.parse_args()


def build_combined_manifest(categories: list) -> pd.DataFrame:
    """Concatenate every category's manifest, using every row (train +
    test) and replacing the good/defective "label" with a category index."""
    categories = sorted(categories)
    frames = []
    for category in categories:
        manifest = load_manifest(PROJECT_ROOT / "data" / "manifests" / f"{category}.csv")
        manifest = manifest[["image_path"]].copy()
        manifest["label"] = categories.index(category)
        frames.append(manifest)
    combined = pd.concat(frames, ignore_index=True)
    return combined, categories


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


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)

    manifest, categories = build_combined_manifest(args.categories)
    print(f"Categories (label order): {categories}")
    print(f"Total images: {len(manifest)}")

    train_subset, val_subset = train_test_split(
        manifest, test_size=args.val_split, random_state=args.seed, stratify=manifest["label"]
    )
    print(f"Train subset: {len(train_subset)} images | Val subset: {len(val_subset)} images")

    train_dataset = ManifestImageDataset(
        train_subset, PROJECT_ROOT, transform=get_train_transforms(args.image_size)
    )
    val_dataset = ManifestImageDataset(
        val_subset, PROJECT_ROOT, transform=get_val_transforms(args.image_size)
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")
    model = build_baseline_model(
        architecture="resnet18", num_classes=len(categories), pretrained=True
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy, y_true, y_pred = evaluate(model, val_loader, criterion, device)
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
            f"Epoch {epoch}/{args.epochs} - "
            f"accuracy: {train_accuracy:.4f} - loss: {train_loss:.4f} - "
            f"val_accuracy: {val_accuracy:.4f} - val_loss: {val_loss:.4f} - "
            f"learning_rate: {learning_rate:.4g}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
                break

    model.load_state_dict(best_state)
    _val_loss, _val_accuracy, y_true, y_pred = evaluate(model, val_loader, criterion, device)

    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred).tolist()
    per_class_report = compute_per_class_report(y_true, y_pred, categories)
    print(f"Validation accuracy (best epoch restored): {accuracy:.4f}")
    print(f"Confusion matrix (rows=true, cols=pred, order={categories}): {cm}")
    print("Per-class report:")
    print(json.dumps(per_class_report, indent=2))

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    METRICS_PATH.write_text(
        json.dumps(
            {
                "categories": categories,
                "val_accuracy": accuracy,
                "confusion_matrix": cm,
                "classification_report": per_class_report,
                "history": history,
                "seed": args.seed,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "epochs_trained": len(history),
                "early_stopping_patience": args.early_stopping_patience,
            },
            indent=2,
        )
    )
    plot_training_curves(history, output_path=FIGURE_PATH, title_prefix="Category classifier")

    print(f"Saved checkpoint to {CHECKPOINT_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")
    print(f"Saved training curves figure to {FIGURE_PATH}")


if __name__ == "__main__":
    main()
