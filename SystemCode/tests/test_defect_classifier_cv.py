import numpy as np

from src.models.summarize_defect_classifier_cv import paired_counts, strategy_probabilities


def _fold(y, convnext_pred, densenet_pred, n_classes=3):
    def onehot(pred):
        return np.eye(n_classes)[pred]

    y = np.asarray(y)
    return 0, y, {
        "single_convnext_tiny": onehot(convnext_pred),
        "single_densenet121": onehot(densenet_pred),
    }


def test_paired_counts_and_four_model_strategies():
    folds = [_fold([0, 1, 2], [0, 1, 0], [0, 0, 2])]
    assert paired_counts(folds, "single_densenet121", "single_convnext_tiny")[:2] == (1, 1)
    probs = {name: np.eye(3)[[0, 1, 2]] for name in ("convnext_tiny", "resnet18", "efficientnet_b0", "densenet121")}
    strategies = strategy_probabilities(probs)
    assert {"four_model_hard_vote", "four_model_equal_soft_vote", "convnext_heavy_soft_vote"} <= set(strategies)
