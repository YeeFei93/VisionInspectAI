import numpy as np

from src.models.benchmark_defect_classifiers import classification_scores, sample_std


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


def test_sample_std_is_zero_for_one_category():
    assert sample_std([0.75]) == 0.0
