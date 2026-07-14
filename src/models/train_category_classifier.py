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
import json
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from src.data.dataset import ManifestImageDataset, load_manifest
from src.models.baseline_classifier import build_baseline_model
from src.preprocessing.transform import get_train_transforms, get_val_transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "checkpoints" / "category_classifier_resnet18.pt"
METRICS_PATH = PROJECT_ROOT / "outputs" / "metrics" / "category_classifier_metrics.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["screw", "bottle", "hazelnut"],
        help="Categories to include (must each have data/manifests/<category>.csv).",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
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


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())
    return all_labels, all_preds


def main() -> None:
    args = parse_args()

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_baseline_model(
        architecture="resnet18", num_classes=len(categories), pretrained=True
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch}/{args.epochs} - train_loss: {train_loss:.4f}")

    y_true, y_pred = predict(model, val_loader, device)
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred).tolist()
    print(f"Validation accuracy: {accuracy:.4f}")
    print(f"Confusion matrix (rows=true, cols=pred, order={categories}): {cm}")

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    METRICS_PATH.write_text(
        json.dumps({"categories": categories, "val_accuracy": accuracy, "confusion_matrix": cm}, indent=2)
    )

    print(f"Saved checkpoint to {CHECKPOINT_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
