from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.benchmark_defect_classifiers import (
    classification_scores,
    expand_training_rows,
    save_deployment_artifacts,
)
from src.models.run_defect_classifier_ensembles import (
    build_ensemble_strategies,
    hard_vote_probabilities,
    stratified_bootstrap_rows,
    weighted_soft_vote,
)
from src.models.train_defect_classifier import build_defect_manifest


def test_build_defect_manifest_rejects_single_defect_type(monkeypatch):
    manifest = pd.DataFrame(
        {
            "split": ["test", "test"],
            "label": [1, 1],
            "defect_type": ["defective", "defective"],
        }
    )
    monkeypatch.setattr(
        "src.models.train_defect_classifier.load_manifest",
        lambda _path: manifest,
    )

    with pytest.raises(ValueError, match="at least two distinct defect types"):
        build_defect_manifest(Path("unused.csv"))


def test_classification_scores_reports_accuracy_and_macro_f1():
    scores = classification_scores(
        np.array([0, 0, 1, 1]),
        np.array(
            [
                [0.9, 0.1],
                [0.8, 0.2],
                [0.7, 0.3],
                [0.6, 0.4],
            ]
        ),
    )

    assert scores["val_accuracy"] == 0.5
    assert scores["val_macro_f1"] == 1 / 3


def test_expand_training_rows_applies_per_class_variants():
    rows = pd.DataFrame(
        {
            "image_path": ["a.png", "b.png"],
            "defect_type": ["bent_lead", "misplaced"],
            "label": [0, 1],
        }
    )

    expanded = expand_training_rows(rows, "transistor")

    assert expanded.groupby("image_path")["augmentation_variant"].apply(list).to_dict() == {
        "a.png": ["identity", "hflip"],
        "b.png": ["identity", "hflip", "vflip", "rot180"],
    }
    assert expanded.loc[expanded["image_path"] == "b.png", "label"].eq(1).all()


def test_expand_training_rows_grows_default_category_sixfold():
    rows = pd.DataFrame({"image_path": ["a.png"], "defect_type": ["scratch"], "label": [0]})

    assert len(expand_training_rows(rows, "screw")) == 6


def test_save_deployment_artifacts_writes_weights_and_label_metadata(tmp_path):
    import json

    import torch

    model = torch.nn.Linear(2, 1)
    checkpoint_path = tmp_path / "checkpoints" / "model.pt"
    metrics_path = tmp_path / "metrics" / "model_metrics.json"
    metadata = {"defect_types": ["scratch"], "crop_mode": "full_image"}

    save_deployment_artifacts(model, checkpoint_path, metrics_path, metadata)

    saved_state = torch.load(checkpoint_path, map_location="cpu")
    saved_metadata = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert set(saved_state) == set(model.state_dict())
    assert saved_metadata == metadata


def test_hard_vote_breaks_ties_by_mean_probability_and_normalizes():
    probabilities = hard_vote_probabilities(
        {
            "first": np.array([[0.9, 0.1]]),
            "second": np.array([[0.2, 0.8]]),
        }
    )

    assert probabilities.argmax(axis=1).tolist() == [0]
    assert probabilities.sum(axis=1).tolist() == [1.0]


def test_weighted_soft_vote_applies_weights():
    probabilities = weighted_soft_vote(
        {
            "convnext": np.array([[0.9, 0.1]]),
            "dense": np.array([[0.1, 0.9]]),
        },
        {"convnext": 0.75, "dense": 0.25},
    )

    np.testing.assert_allclose(probabilities, [[0.7, 0.3]])


def test_weight_sweep_is_limited_to_bottle_and_wood():
    predictions = {
        name: np.array([[0.6, 0.4]])
        for name in ("convnext_tiny", "resnet18", "efficientnet_b0", "densenet121")
    }

    bottle_strategies = build_ensemble_strategies(predictions, predictions["convnext_tiny"], "bottle")
    screw_strategies = build_ensemble_strategies(predictions, predictions["convnext_tiny"], "screw")

    assert "convnext_25_densenet_75" in bottle_strategies
    assert "convnext_50_densenet_50" in bottle_strategies
    assert "convnext_75_densenet_25" in bottle_strategies
    assert not any("densenet_" in name for name in screw_strategies)


def test_stratified_bootstrap_preserves_each_class_count_with_replacement():
    rows = pd.DataFrame(
        {
            "image_path": ["a", "b", "c", "d", "e"],
            "label": [0, 0, 0, 1, 1],
        }
    )

    bootstrapped = stratified_bootstrap_rows(rows, seed=42)

    assert bootstrapped["label"].value_counts().to_dict() == {0: 3, 1: 2}
    assert len(bootstrapped["image_path"].unique()) < len(bootstrapped)
