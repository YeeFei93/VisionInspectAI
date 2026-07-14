"""Unit tests for src/evaluation/metrics.py (pure functions, no I/O)."""

import numpy as np

from src.evaluation.metrics import (
    compute_classification_metrics,
    compute_dice,
    compute_iou,
    compute_pixel_level_metrics,
    youden_threshold,
)


def test_compute_classification_metrics_perfect():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]
    metrics = compute_classification_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]


def test_compute_classification_metrics_all_wrong():
    y_true = [0, 0, 1, 1]
    y_pred = [1, 1, 0, 0]
    metrics = compute_classification_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 0.0
    assert metrics["recall"] == 0.0


def test_youden_threshold_separates_perfectly_separable_scores():
    y_true = [0, 0, 0, 1, 1, 1]
    scores = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    threshold = youden_threshold(y_true, scores)
    preds = [1 if s >= threshold else 0 for s in scores]
    assert preds == y_true


def test_compute_iou_identical_masks_is_one():
    mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    assert compute_iou(mask, mask) == 1.0


def test_compute_iou_disjoint_masks_is_zero():
    pred = np.array([[1, 0], [0, 0]], dtype=np.uint8)
    gt = np.array([[0, 0], [0, 1]], dtype=np.uint8)
    assert compute_iou(pred, gt) == 0.0


def test_compute_iou_both_empty_is_one():
    empty = np.zeros((2, 2), dtype=np.uint8)
    assert compute_iou(empty, empty) == 1.0


def test_compute_dice_identical_masks_is_one():
    mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    assert compute_dice(mask, mask) == 1.0


def test_compute_dice_disjoint_masks_is_zero():
    pred = np.array([[1, 0], [0, 0]], dtype=np.uint8)
    gt = np.array([[0, 0], [0, 1]], dtype=np.uint8)
    assert compute_dice(pred, gt) == 0.0


def test_compute_pixel_level_metrics_perfect_localization():
    # Two 2x2 "images": one all-normal, one with a single defective pixel
    # that the anomaly map scores clearly higher than everything else.
    anomaly_maps = [
        np.array([[0.1, 0.1], [0.1, 0.1]]),
        np.array([[0.1, 0.1], [0.1, 0.9]]),
    ]
    gt_masks = [
        np.zeros((2, 2), dtype=np.uint8),
        np.array([[0, 0], [0, 1]], dtype=np.uint8),
    ]
    result = compute_pixel_level_metrics(anomaly_maps, gt_masks)
    assert result["pixel_auroc"] == 1.0
    assert result["mean_iou"] == 1.0
    assert result["mean_dice"] == 1.0
