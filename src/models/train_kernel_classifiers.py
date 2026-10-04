"""Train RBF, GRNN, and SOM models on fixed ResNet18 image embeddings.

Examples:
    python -m src.models.train_kernel_classifiers --task baseline --config config/screw_config.yaml
    python -m src.models.train_kernel_classifiers --task category --categories screw bottle hazelnut carpet leather grid tile wood
    python -m src.models.train_kernel_classifiers --task defect --config config/screw_config.yaml
    python -m src.models.train_kernel_classifiers --task anomaly --config config/screw_config.yaml
"""

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.data.dataset import ManifestImageDataset, load_manifest, make_train_val_split
from src.evaluation.metrics import compute_classification_metrics, plot_confusion_matrix, youden_threshold
from src.models.kernel_classifiers import GRNNClassifier, RBFNetworkClassifier, ResNet18EmbeddingExtractor, SOMClassifier
from src.preprocessing.transform import get_val_transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("baseline", "category", "defect", "anomaly"), required=True)
    parser.add_argument("--config", type=Path, default=Path("config/screw_config.yaml"))
    parser.add_argument("--categories", nargs="+", default=["screw", "bottle", "hazelnut", "carpet", "leather", "grid", "tile", "wood"])
    parser.add_argument("--classifier", choices=("rbf", "grnn", "som", "both"), default="both")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-centers", type=int, default=64)
    parser.add_argument("--grnn-bandwidth", type=float, default=1.0)
    parser.add_argument("--som-rows", type=int, default=4)
    parser.add_argument("--som-cols", type=int, default=4)
    parser.add_argument("--som-epochs", type=int, default=40)
    parser.add_argument("--som-learning-rate", type=float, default=0.4)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    with config_path.open() as file:
        return yaml.safe_load(file)


def build_category_manifest(categories: list[str]) -> tuple[pd.DataFrame, list[str]]:
    categories = sorted(categories)
    frames = []
    for category in categories:
        manifest = load_manifest(PROJECT_ROOT / "data" / "manifests" / f"{category}.csv")
        frame = manifest[["image_path"]].copy()
        frame["label"] = categories.index(category)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True), categories


def build_defect_manifest(manifest_path: Path) -> tuple[pd.DataFrame, list[str]]:
    manifest = load_manifest(manifest_path)
    defective = manifest[(manifest["split"] == "test") & (manifest["label"] == 1)].copy()
    defect_types = sorted(defective["defect_type"].unique())
    defective["label"] = defective["defect_type"].map(defect_types.index)
    return defective.reset_index(drop=True), defect_types


def make_task_split(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, str, list[str]]:
    if args.task == "baseline":
        config = load_config(args.config)
        manifest = load_manifest(PROJECT_ROOT / config["data"]["manifest_path"])
        train_subset, val_subset = make_train_val_split(
            manifest, val_split=config["data"]["val_split"], seed=config["data"]["seed"]
        )
        return train_subset, val_subset, config.get("category", "screw"), ["good", "defective"]
    if args.task == "category":
        manifest, class_names = build_category_manifest(args.categories)
        train_subset, val_subset = train_test_split(
            manifest, test_size=0.2, random_state=42, stratify=manifest["label"]
        )
        return train_subset, val_subset, "all", class_names
    config = load_config(args.config)
    manifest, class_names = build_defect_manifest(PROJECT_ROOT / config["data"]["manifest_path"])
    train_subset, val_subset = train_test_split(
        manifest,
        test_size=config["data"]["val_split"],
        random_state=config["data"]["seed"],
        stratify=manifest["label"],
    )
    return train_subset, val_subset, config.get("category", "screw"), class_names


def build_loader(manifest: pd.DataFrame, batch_size: int, image_size: int) -> DataLoader:
    dataset = ManifestImageDataset(manifest, PROJECT_ROOT, transform=get_val_transforms(image_size))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def make_classifier(name: str, args: argparse.Namespace):
    if name == "rbf":
        return RBFNetworkClassifier(n_centers=args.num_centers)
    if name == "grnn":
        return GRNNClassifier(bandwidth=args.grnn_bandwidth)
    return SOMClassifier(
        map_rows=args.som_rows,
        map_cols=args.som_cols,
        epochs=args.som_epochs,
        learning_rate=args.som_learning_rate,
    )


def plot_som_u_matrix(som: SOMClassifier, output_path: Path) -> None:
    figure, axes = plt.subplots(figsize=(5, 4))
    image = axes.imshow(som.u_matrix(), cmap="viridis")
    axes.set_title("SOM U-Matrix")
    axes.set_xlabel("Map column")
    axes.set_ylabel("Map row")
    figure.colorbar(image, ax=axes, label="Mean neighbor distance")
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def compute_metrics(y_true, y_pred, probabilities, class_names: list[str]) -> dict:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(class_names)))).tolist(),
        "class_names": class_names,
    }
    auc_key = "roc_auc_ovr_macro" if len(class_names) > 2 else "roc_auc"
    auc_scores = []
    for class_index in range(len(class_names)):
        binary_labels = (np.asarray(y_true) == class_index).astype(int)
        if len(np.unique(binary_labels)) == 2:
            auc_scores.append(roc_auc_score(binary_labels, probabilities[:, class_index]))
    metrics[auc_key] = float(np.mean(auc_scores)) if len(auc_scores) == len(class_names) else None
    return metrics


def main() -> None:
    args = parse_args()
    if args.task == "anomaly":
        run_anomaly_exploration(args)
        return
    train_subset, val_subset, scope, class_names = make_task_split(args)
    config = load_config(args.config) if args.task != "category" else {}
    image_size = config.get("data", {}).get("image_size", 224)
    train_loader = build_loader(train_subset, args.batch_size, image_size)
    val_loader = build_loader(val_subset, args.batch_size, image_size)
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    extractor = ResNet18EmbeddingExtractor(device, pretrained=not args.no_pretrained)
    train_features, train_labels = extractor.transform(train_loader)
    val_features, val_labels = extractor.transform(val_loader)
    classifier_names = ("rbf", "grnn", "som") if args.classifier == "both" else (args.classifier,)

    checkpoint_dir = PROJECT_ROOT / "models" / "checkpoints"
    metrics_dir = PROJECT_ROOT / "outputs" / "metrics"
    figures_dir = PROJECT_ROOT / "outputs" / "figures"
    for directory in (checkpoint_dir, metrics_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    for classifier_name in classifier_names:
        classifier = make_classifier(classifier_name, args).fit(train_features, train_labels)
        probabilities = classifier.predict_proba(val_features)
        predictions = classifier.classes_[probabilities.argmax(axis=1)]
        metrics = compute_metrics(val_labels, predictions, probabilities, class_names)
        run_name = f"{args.task}_{classifier_name}_resnet18_{scope}"
        with (checkpoint_dir / f"{run_name}.pkl").open("wb") as file:
            pickle.dump(classifier, file)
        (metrics_dir / f"{run_name}_metrics.json").write_text(json.dumps(metrics, indent=2))
        plot_confusion_matrix(
            val_labels,
            predictions,
            class_names=class_names,
            output_path=figures_dir / f"{run_name}_confusion_matrix.png",
            title=f"{classifier_name} — confusion matrix",
        )
        if isinstance(classifier, SOMClassifier):
            plot_som_u_matrix(classifier, figures_dir / f"{run_name}_u_matrix.png")
        print(json.dumps({"model": run_name, **metrics}, indent=2))


def run_anomaly_exploration(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    manifest = load_manifest(PROJECT_ROOT / config["data"]["manifest_path"])
    train_subset = manifest[(manifest["split"] == "train") & (manifest["label"] == 0)].copy()
    test_subset = manifest[manifest["split"] == "test"].copy()
    image_size = config["data"]["image_size"]
    train_loader = build_loader(train_subset, args.batch_size, image_size)
    test_loader = build_loader(test_subset, args.batch_size, image_size)
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    extractor = ResNet18EmbeddingExtractor(device, pretrained=not args.no_pretrained)
    train_features, _ = extractor.transform(train_loader)
    test_features, test_labels = extractor.transform(test_loader)
    som = make_classifier("som", args).fit(train_features)
    scores = som.quantization_error(test_features)
    threshold = youden_threshold(test_labels, scores)
    predictions = (scores >= threshold).astype(int)
    metrics = compute_classification_metrics(test_labels, predictions)
    metrics.update({"roc_auc": float(roc_auc_score(test_labels, scores)), "threshold": threshold})
    patchcore_metrics_path = PROJECT_ROOT / config["output"]["metrics_dir"] / (
        f"patchcore_{config['anomaly_detection']['backbone']}_{config.get('category', 'screw')}_metrics.json"
    )
    if patchcore_metrics_path.exists():
        patchcore_metrics = json.loads(patchcore_metrics_path.read_text())
        metrics["patchcore_roc_auc"] = patchcore_metrics.get("auroc")

    checkpoint_dir = PROJECT_ROOT / config["output"]["checkpoint_dir"]
    metrics_dir = PROJECT_ROOT / config["output"]["metrics_dir"]
    figures_dir = PROJECT_ROOT / config["output"]["figures_dir"]
    for directory in (checkpoint_dir, metrics_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)
    category = config.get("category", "screw")
    run_name = f"som_anomaly_resnet18_{category}"
    with (checkpoint_dir / f"{run_name}.pkl").open("wb") as file:
        pickle.dump(som, file)
    (metrics_dir / f"{run_name}_metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_som_u_matrix(som, figures_dir / f"{run_name}_u_matrix.png")
    print(json.dumps({"model": run_name, **metrics}, indent=2))


if __name__ == "__main__":
    main()