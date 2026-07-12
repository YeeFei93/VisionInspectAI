"""Metrics for classification (baseline classifier, image-level anomaly
score) and pixel-level defect localization (anomaly heatmap vs
ground-truth mask)."""

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_classification_metrics(
    y_true: Sequence[int], y_pred: Sequence[int]
) -> dict:
    """Accuracy, precision, recall, F1 (positive class = defective/1), and the
    raw confusion matrix, in the format described in the project spec."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def youden_threshold(y_true: Sequence[int], scores: Sequence[float]) -> float:
    """Threshold that maximizes Youden's J statistic (TPR - FPR) on the ROC
    curve. Used to turn continuous anomaly scores into good/defective
    decisions at either the image level or the pixel level."""
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    j_scores = tpr - fpr
    return float(thresholds[np.argmax(j_scores)])


def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Intersection-over-Union between a binary predicted mask and the
    binary ground-truth defect mask."""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def compute_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Dice coefficient (F1 over pixels) between a binary predicted mask and
    the binary ground-truth defect mask."""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    denom = pred_mask.sum() + gt_mask.sum()
    if denom == 0:
        return 1.0
    return float(2 * intersection / denom)


def compute_pixel_level_metrics(
    anomaly_maps: Sequence[np.ndarray],
    gt_masks: Sequence[np.ndarray],
) -> dict:
    """Pixel-level AUROC over every test pixel (does the heatmap highlight
    true defect pixels?), plus mean IoU and mean Dice at the Youden's-J
    pixel threshold (how much does the predicted defect area overlap the
    ground-truth mask?). `gt_masks` should include all-zero masks for
    "good" images so the AUROC is computed over the full test set.
    """
    pixel_scores = np.concatenate([m.ravel() for m in anomaly_maps])
    pixel_labels = np.concatenate([m.ravel() for m in gt_masks]).astype(int)

    pixel_auroc = roc_auc_score(pixel_labels, pixel_scores)
    pixel_threshold = youden_threshold(pixel_labels, pixel_scores)

    iou_scores, dice_scores = [], []
    for anomaly_map, gt_mask in zip(anomaly_maps, gt_masks):
        if gt_mask.sum() == 0:
            continue  # IoU/Dice are only meaningful for images with a real defect mask
        pred_mask = (anomaly_map >= pixel_threshold).astype(np.uint8)
        iou_scores.append(compute_iou(pred_mask, gt_mask))
        dice_scores.append(compute_dice(pred_mask, gt_mask))

    return {
        "pixel_auroc": float(pixel_auroc),
        "pixel_threshold": float(pixel_threshold),
        "mean_iou": float(np.mean(iou_scores)) if iou_scores else None,
        "mean_dice": float(np.mean(dice_scores)) if dice_scores else None,
    }


def plot_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str] = ("good", "defective"),
    output_path: Optional[Path] = None,
):
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(4, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Baseline classifier — confusion matrix")
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path)

    return fig
