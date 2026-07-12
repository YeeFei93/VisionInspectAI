"""Train the first baseline supervised good-vs-defective classifier.

The MVTec-AD train/ folder only contains "good" images, so this baseline
instead splits the labeled test/ rows (good + every defect type) into a
train subset and a held-out validation subset. This is a supervised
baseline only — the proper approach for this dataset is unsupervised
anomaly detection trained on train/good (see src/models/anomaly_detector.py).

Usage:
    python -m src.models.train_baseline --config config/screw_config.yaml
"""

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from src.data.dataset import ManifestImageDataset, load_manifest, make_train_val_split
from src.evaluation.metrics import compute_classification_metrics, plot_confusion_matrix
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

    manifest = load_manifest(PROJECT_ROOT / data_cfg["manifest_path"])
    train_subset, val_subset = make_train_val_split(
        manifest, val_split=data_cfg["val_split"], seed=data_cfg["seed"]
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_baseline_model(
        architecture=model_cfg["architecture"],
        num_classes=model_cfg["num_classes"],
        pretrained=model_cfg["pretrained"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])

    for epoch in range(1, train_cfg["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch}/{train_cfg['epochs']} - train_loss: {train_loss:.4f}")

    y_true, y_pred = predict(model, val_loader, device)
    metrics = compute_classification_metrics(y_true, y_pred)
    print("Validation metrics:")
    print(json.dumps(metrics, indent=2))

    checkpoint_dir = PROJECT_ROOT / output_cfg["checkpoint_dir"]
    metrics_dir = PROJECT_ROOT / output_cfg["metrics_dir"]
    figures_dir = PROJECT_ROOT / output_cfg["figures_dir"]
    for directory in (checkpoint_dir, metrics_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    run_name = f"baseline_{model_cfg['architecture']}_{category}"

    checkpoint_path = checkpoint_dir / f"{run_name}.pt"
    torch.save(model.state_dict(), checkpoint_path)

    metrics_path = metrics_dir / f"{run_name}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    figure_path = figures_dir / f"{run_name}_confusion_matrix.png"
    plot_confusion_matrix(y_true, y_pred, output_path=figure_path)

    print(f"Saved checkpoint to {checkpoint_path}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved confusion matrix figure to {figure_path}")


if __name__ == "__main__":
    main()
