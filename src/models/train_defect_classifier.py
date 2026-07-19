"""Train a per-category "what kind of defect is this?" classifier.

Unlike the baseline good-vs-defective classifier, this only looks at the
*defective* rows of one category's manifest (test/ split, label == 1) and
predicts which defect_type it is (e.g. leather: color / cut / fold / glue /
poke). Used by the Streamlit demo to show a defect type alongside the
Normal/Defective verdict and severity.

Usage:
    python -m src.models.train_defect_classifier --config config/leather_config.yaml
"""

import argparse
import json
from pathlib import Path

import torch
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader

from src.data.dataset import ManifestImageDataset, load_manifest
from src.models.baseline_classifier import build_baseline_model
from src.preprocessing.transform import get_train_transforms, get_val_transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("config/screw_config.yaml")
    )
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
    config = load_config(args.config)

    category = config.get("category", "screw")
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["train"]
    output_cfg = config["output"]

    manifest, defect_types = build_defect_manifest(PROJECT_ROOT / data_cfg["manifest_path"])
    print(f"Category: {category} | Defect types (label order): {defect_types}")
    print(f"Total defective images: {len(manifest)}")

    train_subset, val_subset = train_test_split(
        manifest, test_size=data_cfg["val_split"], random_state=data_cfg["seed"], stratify=manifest["label"]
    )
    print(f"Train subset: {len(train_subset)} images | Val subset: {len(val_subset)} images")

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

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")
    model = build_baseline_model(
        architecture=model_cfg["architecture"],
        num_classes=len(defect_types),
        pretrained=model_cfg["pretrained"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])

    for epoch in range(1, train_cfg["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch}/{train_cfg['epochs']} - train_loss: {train_loss:.4f}")

    y_true, y_pred = predict(model, val_loader, device)
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred).tolist()
    print(f"Validation accuracy: {accuracy:.4f}")
    print(f"Confusion matrix (rows=true, cols=pred, order={defect_types}): {cm}")

    checkpoint_dir = PROJECT_ROOT / output_cfg["checkpoint_dir"]
    metrics_dir = PROJECT_ROOT / output_cfg["metrics_dir"]
    for directory in (checkpoint_dir, metrics_dir):
        directory.mkdir(parents=True, exist_ok=True)

    run_name = f"defect_classifier_{model_cfg['architecture']}_{category}"

    checkpoint_path = checkpoint_dir / f"{run_name}.pt"
    torch.save(model.state_dict(), checkpoint_path)

    metrics_path = metrics_dir / f"{run_name}_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {"defect_types": defect_types, "val_accuracy": accuracy, "confusion_matrix": cm}, indent=2
        )
    )

    print(f"Saved checkpoint to {checkpoint_path}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
