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
import copy
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
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

    train_subset, val_subset = train_test_split(
        manifest, test_size=data_cfg["val_split"], random_state=data_cfg["seed"], stratify=manifest["label"]
    )
    print(f"Train subset: {len(train_subset)} images | Val subset: {len(val_subset)} images")

    image_size = data_cfg["image_size"]
    crop_kwargs = {
        "focus_on_mask": args.defect_focused_crops,
        "crop_padding_ratio": args.crop_padding_ratio,
        "min_crop_fraction": args.min_crop_fraction,
    }
    train_dataset = ManifestImageDataset(
        train_subset,
        PROJECT_ROOT,
        transform=get_train_transforms(image_size),
        **crop_kwargs,
    )
    val_dataset = ManifestImageDataset(
        val_subset,
        PROJECT_ROOT,
        transform=get_val_transforms(image_size),
        **crop_kwargs,
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
    if args.freeze_backbone:
        model = freeze_backbone(model, model_cfg["architecture"])
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Backbone frozen; trainable parameters: {trainable}")

    criterion = nn.CrossEntropyLoss()
    learning_rate_cfg = args.learning_rate if args.learning_rate is not None else train_cfg["learning_rate"]
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=learning_rate_cfg
    )
    patience = int(train_cfg.get("early_stopping_patience", 3))
    min_delta = float(train_cfg.get("early_stopping_min_delta", 0.0))

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    epochs = args.epochs if args.epochs is not None else train_cfg["epochs"]
    for epoch in range(1, epochs + 1):
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
            f"Epoch {epoch}/{epochs} - "
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
    _val_loss, _val_accuracy, y_true, y_pred = evaluate(model, val_loader, criterion, device)

    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred).tolist()
    per_class_report = compute_per_class_report(y_true, y_pred, defect_types)
    print(f"Validation accuracy (best epoch restored): {accuracy:.4f}")
    print(f"Confusion matrix (rows=true, cols=pred, order={defect_types}): {cm}")
    print("Per-class report:")
    print(json.dumps(per_class_report, indent=2))

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
                "history": history,
                "seed": data_cfg["seed"],
                "freeze_backbone": args.freeze_backbone,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "epochs_trained": len(history),
                "early_stopping_patience": patience,
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
    plot_training_curves(history, output_path=figure_path, title_prefix="Defect-type classifier")

    print(f"Saved checkpoint to {checkpoint_path}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved training curves figure to {figure_path}")


if __name__ == "__main__":
    main()
