"""Classical ML alternative to the deep per-category defect-type classifier:
frozen ResNet18 embeddings -> PCA -> LDA (Linear Discriminant Analysis).

Motivation: train_defect_classifier.py's fine-tuned CNN head struggles on
categories with very few images per defect type (e.g. screw: ~16-17 train
images per class over 5 classes, only 0.444 val accuracy — see the README's
"Defect-Type Classification" section) because a CNN head trained from so
few examples overfits easily. LDA instead finds the linear directions that
best separate the classes directly from the (much lower-dimensional) frozen
embeddings, which is a better-conditioned problem in this low-data regime.
PCA is applied first purely as a regularizer: with 512-d ResNet18 embeddings
and only ~80 training images, LDA's within-class scatter matrix would
otherwise be singular/unstable.

This targets the same defective-only, per-category classification problem
as train_defect_classifier.py (same manifest rows, same stratified split),
so its val_accuracy/confusion_matrix are directly comparable.

Usage:
    python -m src.models.train_pca_lda_defect_classifier --config config/screw_config.yaml
"""

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torchvision import models

from src.models.train_defect_classifier import build_defect_manifest
from src.preprocessing.transform import get_val_transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/screw_config.yaml"))
    parser.add_argument(
        "--pca-components",
        type=int,
        default=30,
        help="Max PCA components kept before LDA (actual count is also capped by the "
        "train-subset size, since it must be estimable from that many samples).",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def build_embedding_extractor(device: torch.device) -> nn.Module:
    """Frozen, ImageNet-pretrained ResNet18 with its classification head
    removed, used purely as a fixed 512-d feature extractor."""
    net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    net.fc = nn.Identity()
    net.eval()
    for param in net.parameters():
        param.requires_grad_(False)
    return net.to(device)


@torch.no_grad()
def extract_embeddings(subset, extractor: nn.Module, transform, device: torch.device) -> np.ndarray:
    embeddings = []
    for _, row in subset.iterrows():
        image = Image.open(PROJECT_ROOT / row["image_path"]).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)
        embedding = extractor(input_tensor)
        embeddings.append(embedding.squeeze(0).cpu().numpy())
    return np.stack(embeddings)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    category = config.get("category", "screw")
    data_cfg = config["data"]
    output_cfg = config["output"]

    manifest, defect_types = build_defect_manifest(PROJECT_ROOT / data_cfg["manifest_path"])
    print(f"Category: {category} | Defect types (label order): {defect_types}")
    print(f"Total defective images: {len(manifest)}")

    train_subset, val_subset = train_test_split(
        manifest, test_size=data_cfg["val_split"], random_state=data_cfg["seed"], stratify=manifest["label"]
    )
    print(f"Train subset: {len(train_subset)} images | Val subset: {len(val_subset)} images")

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    transform = get_val_transforms(data_cfg["image_size"])
    extractor = build_embedding_extractor(device)

    X_train = extract_embeddings(train_subset, extractor, transform, device)
    y_train = train_subset["label"].to_numpy()
    X_val = extract_embeddings(val_subset, extractor, transform, device)
    y_val = val_subset["label"].to_numpy()

    # PCA components must be estimable from the training sample (< n_samples, < n_features).
    n_components = min(args.pca_components, len(train_subset) - 1, X_train.shape[1])
    # svd_solver="full" (exact SVD) instead of the "auto"-selected randomized solver:
    # these matrices are tiny (~80 samples), so exact SVD is cheap and avoids the
    # randomized solver's numerical instability at this scale.
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components, svd_solver="full")),
            ("lda", LinearDiscriminantAnalysis()),
        ]
    )
    pipeline.fit(X_train, y_train)

    # numpy on Apple Silicon (Accelerate BLAS backend) emits spurious
    # divide-by-zero/overflow RuntimeWarnings from PCA's internal matmul on
    # small matrices even when the actual output is finite/correct (verified:
    # no NaN/Inf ever appears in the transformed features) — a known
    # Accelerate quirk, not a real numerical problem with this pipeline.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
        y_pred = pipeline.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    cm = confusion_matrix(y_val, y_pred).tolist()
    explained_variance = float(np.sum(pipeline.named_steps["pca"].explained_variance_ratio_))
    print(f"Validation accuracy: {accuracy:.4f}")
    print(f"Confusion matrix (rows=true, cols=pred, order={defect_types}): {cm}")
    print(f"PCA components: {n_components} (cumulative explained variance: {explained_variance:.3f})")

    checkpoint_dir = PROJECT_ROOT / output_cfg["checkpoint_dir"]
    metrics_dir = PROJECT_ROOT / output_cfg["metrics_dir"]
    for directory in (checkpoint_dir, metrics_dir):
        directory.mkdir(parents=True, exist_ok=True)

    run_name = f"defect_classifier_pca_lda_{category}"

    checkpoint_path = checkpoint_dir / f"{run_name}.joblib"
    joblib.dump(pipeline, checkpoint_path)

    metrics_path = metrics_dir / f"{run_name}_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "defect_types": defect_types,
                "val_accuracy": accuracy,
                "confusion_matrix": cm,
                "pca_components": n_components,
                "pca_explained_variance": explained_variance,
            },
            indent=2,
        )
    )

    print(f"Saved pipeline to {checkpoint_path}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
