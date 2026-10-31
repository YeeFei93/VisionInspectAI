"""Classical ML alternative to the deep baseline good-vs-defective classifier:
frozen ResNet18 embeddings -> PCA -> LDA (Linear Discriminant Analysis).

Same motivation and same train/val split as train_pca_lda_defect_classifier.py
(see its docstring), applied to the *coarser* good-vs-defective problem
instead of fine-grained defect types — targets the same manifest split as
train_baseline.py (src/data/dataset.py's make_train_val_split) so its
val metrics are directly comparable to the Model Comparison table in
README.md.

Usage:
    python -m src.models.train_pca_lda_baseline_classifier --config config/screw_config.yaml
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
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.dataset import load_manifest, make_train_val_split
from src.evaluation.metrics import compute_classification_metrics
from src.models.train_pca_lda_defect_classifier import build_embedding_extractor, extract_embeddings
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


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    category = config.get("category", "screw")
    data_cfg = config["data"]
    output_cfg = config["output"]

    manifest = load_manifest(PROJECT_ROOT / data_cfg["manifest_path"])
    train_subset, val_subset = make_train_val_split(
        manifest, val_split=data_cfg["val_split"], seed=data_cfg["seed"]
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
    # avoids the randomized solver's numerical instability at this scale (see
    # train_pca_lda_defect_classifier.py's docstring/notes for the same reasoning).
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components, svd_solver="full")),
            ("lda", LinearDiscriminantAnalysis()),
        ]
    )
    pipeline.fit(X_train, y_train)

    # See train_pca_lda_defect_classifier.py: numpy's Accelerate BLAS backend on
    # Apple Silicon emits spurious (verified harmless) RuntimeWarnings here.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
        y_pred = pipeline.predict(X_val)

    metrics = compute_classification_metrics(y_val.tolist(), y_pred.tolist())
    metrics["pca_components"] = n_components
    metrics["pca_explained_variance"] = float(np.sum(pipeline.named_steps["pca"].explained_variance_ratio_))
    print("Validation metrics:")
    print(json.dumps(metrics, indent=2))

    checkpoint_dir = PROJECT_ROOT / output_cfg["checkpoint_dir"]
    metrics_dir = PROJECT_ROOT / output_cfg["metrics_dir"]
    for directory in (checkpoint_dir, metrics_dir):
        directory.mkdir(parents=True, exist_ok=True)

    run_name = f"baseline_pca_lda_{category}"

    checkpoint_path = checkpoint_dir / f"{run_name}.joblib"
    joblib.dump(pipeline, checkpoint_path)

    metrics_path = metrics_dir / f"{run_name}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f"Saved pipeline to {checkpoint_path}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
