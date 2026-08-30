"""Metrics for classification (baseline classifier, image-level anomaly
score) and pixel-level defect localization (anomaly heatmap vs
ground-truth mask)."""

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_classification_metrics(
    y_true: Sequence[int], y_pred: Sequence[int]
) -> dict:
    """Accuracy, precision, recall, F1 (positive class = defective/1), and the
    raw confusion matrix, in the format described in the project spec.
    Also reports balanced accuracy and Matthews correlation coefficient (MCC),
    which stay honest under the class imbalance most MVTec test splits have
    (far more defective than good images) where plain accuracy can mislead."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def compute_per_class_report(
    y_true: Sequence[int], y_pred: Sequence[int], class_names: Sequence[str]
) -> dict:
    """Per-class precision/recall/F1/support (plus macro/weighted averages),
    for multi-class classifiers (category classifier, defect-type
    classifier) where the aggregate accuracy + confusion matrix alone don't
    show which specific classes are being confused."""
    return classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )


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


def compute_aupro(
    anomaly_maps: Sequence[np.ndarray],
    gt_masks: Sequence[np.ndarray],
    num_thresholds: int = 10,
) -> Optional[float]:
    """Simplified AUPRO (per-region overlap averaged across thresholds and
    ground-truth defect regions): for each of `num_thresholds` evenly spaced
    thresholds, binarize each anomaly map and measure what fraction of every
    individual ground-truth defect region (found via connected-component
    labeling) is covered, then average over all regions and thresholds.
    Unlike plain pixel AUROC/IoU/Dice, this credits a detector for hitting
    every distinct defect region rather than being dominated by the largest
    one -- the standard MVTec AD localization metric's intent, without the
    official FPR-truncated integral."""
    from scipy import ndimage

    defective = [(m, gt) for m, gt in zip(anomaly_maps, gt_masks) if gt.sum() > 0]
    if not defective:
        return None

    all_scores = np.concatenate([m.ravel() for m, _ in defective])
    lo, hi = np.percentile(all_scores, 50), all_scores.max()
    if hi <= lo:
        return None
    thresholds = np.linspace(lo, hi, num_thresholds)

    pro_values = []
    for threshold in thresholds:
        overlaps = []
        for anomaly_map, gt_mask in defective:
            pred_mask = anomaly_map >= threshold
            labeled, num_regions = ndimage.label(gt_mask)
            for region_id in range(1, num_regions + 1):
                region = labeled == region_id
                region_size = region.sum()
                if region_size == 0:
                    continue
                overlaps.append(float(np.logical_and(region, pred_mask).sum()) / region_size)
        if overlaps:
            pro_values.append(float(np.mean(overlaps)))

    return float(np.mean(pro_values)) if pro_values else None


def compute_pixel_level_metrics(
    anomaly_maps: Sequence[np.ndarray],
    gt_masks: Sequence[np.ndarray],
) -> dict:
    """Pixel-level AUROC over every test pixel (does the heatmap highlight
    true defect pixels?), plus mean IoU and mean Dice at the Youden's-J
    pixel threshold (how much does the predicted defect area overlap the
    ground-truth mask?), and AUPRO (does every distinct defect region get
    covered, not just the largest one?). `gt_masks` should include all-zero
    masks for "good" images so the AUROC is computed over the full test set.
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
        "aupro": compute_aupro(anomaly_maps, gt_masks),
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


def plot_score_distribution(
    scores: Sequence[float],
    y_true: Sequence[int],
    threshold: float,
    output_path: Optional[Path] = None,
    title_prefix: str = "PatchCore",
):
    """Histogram of image-level anomaly scores for good vs. defective test
    images, with the chosen decision threshold marked. PatchCore has no
    training loop, so this (rather than a loss/accuracy curve) is the
    diagnostic plot for how well normal/anomalous scores are separated."""
    import matplotlib.pyplot as plt

    scores = np.asarray(scores)
    y_true = np.asarray(y_true)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(scores[y_true == 0], bins=30, alpha=0.6, label="good", color="tab:blue")
    ax.hist(scores[y_true == 1], bins=30, alpha=0.6, label="defective", color="tab:red")
    ax.axvline(threshold, color="black", linestyle="--", label=f"threshold={threshold:.2f}")
    ax.set_xlabel("Anomaly score")
    ax.set_ylabel("Count")
    ax.set_title(f"{title_prefix} — score distribution")
    ax.legend()
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path)

    return fig


def plot_generalization_gap(
    bank_scores: Sequence[float],
    holdout_scores: Sequence[float],
    output_path: Optional[Path] = None,
    title_prefix: str = "PatchCore",
):
    """Compare image-level anomaly scores for train/good images that
    contributed to the memory bank vs. a held-out slice of train/good that
    never did. A held-out distribution shifted well above the memory-bank
    distribution indicates the bank overfits to the specific images it was
    built from rather than generalizing to unseen normal variation."""
    import matplotlib.pyplot as plt

    bank_scores = np.asarray(bank_scores)
    holdout_scores = np.asarray(holdout_scores)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(bank_scores, bins=30, alpha=0.6, label="memory-bank train/good", color="tab:green")
    ax.hist(holdout_scores, bins=30, alpha=0.6, label="held-out train/good", color="tab:orange")
    ax.set_xlabel("Anomaly score")
    ax.set_ylabel("Count")
    ax.set_title(f"{title_prefix} — memory-bank vs. held-out normal generalization")
    ax.legend()
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path)

    return fig


def plot_four_way_distribution(
    bank_scores: Sequence[float],
    holdout_scores: Sequence[float],
    test_good_scores: Sequence[float],
    test_defective_scores: Sequence[float],
    threshold: float,
    output_path: Optional[Path] = None,
    title_prefix: str = "PatchCore",
):
    """Overlay four anomaly-score distributions on one plot: memory-bank
    train/good, held-out train/good, labeled test/good, and test/defective.
    A healthy detector shows all three normal groups (bank/holdout/test-good)
    clustered together, clearly separated from defective. If held-out and/or
    test-good drift noticeably toward the defective group while bank stays
    low, the memory bank is overfitting to the specific images it was built
    from rather than the true normal manifold."""
    import matplotlib.pyplot as plt

    groups = [
        ("memory-bank train/good", bank_scores, "tab:green"),
        ("held-out train/good", holdout_scores, "tab:orange"),
        ("test/good", test_good_scores, "tab:blue"),
        ("test/defective", test_defective_scores, "tab:red"),
    ]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, group_scores, color in groups:
        group_scores = np.asarray(group_scores)
        ax.hist(group_scores, bins=30, alpha=0.5, label=label, color=color)
    ax.axvline(threshold, color="black", linestyle="--", label=f"threshold={threshold:.2f}")
    ax.set_xlabel("Anomaly score")
    ax.set_ylabel("Count")
    ax.set_title(f"{title_prefix} — bank vs. held-out vs. test distributions")
    ax.legend(fontsize=8)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path)

    return fig


def plot_memory_bank_pca(
    memory_bank: np.ndarray,
    output_path: Optional[Path] = None,
    title_prefix: str = "PatchCore",
    max_points: int = 5000,
    seed: int = 42,
):
    """2D PCA scatter of the memory-bank's raw (non-projected) patch
    features -- a qualitative "what does normal look like" visualization
    of the coreset, distinct from the coreset-selection projection used
    internally to pick which patches to keep. Subsamples to `max_points`
    for a fast, readable scatter on large memory banks."""
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    memory_bank = np.asarray(memory_bank)
    if memory_bank.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        indices = rng.choice(memory_bank.shape[0], max_points, replace=False)
        memory_bank = memory_bank[indices]

    reduced = PCA(n_components=2, random_state=seed).fit_transform(memory_bank)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(reduced[:, 0], reduced[:, 1], alpha=0.5, s=8, color="tab:blue")
    ax.set_xlabel("Principal component 1")
    ax.set_ylabel("Principal component 2")
    ax.set_title(f"{title_prefix} — memory-bank feature space (PCA)")
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path)

    return fig


def plot_threshold_sweep(
    scores: Sequence[float],
    y_true: Sequence[int],
    threshold: float,
    output_path: Optional[Path] = None,
    title_prefix: str = "PatchCore",
    num_thresholds: int = 100,
):
    """Accuracy/precision/recall/F1 swept over candidate thresholds spanning
    the observed score range, with the chosen Youden's-J threshold marked.
    Stands in for a per-epoch metric curve since PatchCore fits in one shot
    rather than training over epochs."""
    import matplotlib.pyplot as plt

    scores = np.asarray(scores)
    y_true = np.asarray(y_true)
    candidate_thresholds = np.linspace(scores.min(), scores.max(), num_thresholds)

    accuracies, precisions, recalls, f1s = [], [], [], []
    for candidate in candidate_thresholds:
        y_pred = (scores >= candidate).astype(int)
        accuracies.append(accuracy_score(y_true, y_pred))
        precisions.append(precision_score(y_true, y_pred, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, zero_division=0))
        f1s.append(f1_score(y_true, y_pred, zero_division=0))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(candidate_thresholds, accuracies, label="accuracy")
    ax.plot(candidate_thresholds, precisions, label="precision")
    ax.plot(candidate_thresholds, recalls, label="recall")
    ax.plot(candidate_thresholds, f1s, label="f1")
    ax.axvline(threshold, color="black", linestyle="--", label=f"chosen threshold={threshold:.2f}")
    ax.set_xlabel("Anomaly score threshold")
    ax.set_ylabel("Metric value")
    ax.set_title(f"{title_prefix} — metric vs. threshold")
    ax.legend()
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path)

    return fig


def plot_training_curves(
    history: Sequence[dict],
    output_path: Optional[Path] = None,
    title_prefix: str = "Baseline classifier",
):
    """Per-epoch train/val loss and accuracy curves from a `history` list of
    `{"epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"}`
    dicts (see train_baseline.py)."""
    import matplotlib.pyplot as plt

    epochs = [entry["epoch"] for entry in history]
    train_loss = [entry["train_loss"] for entry in history]
    val_loss = [entry["val_loss"] for entry in history]
    train_accuracy = [entry["train_accuracy"] for entry in history]
    val_accuracy = [entry["val_accuracy"] for entry in history]

    fig, (loss_ax, acc_ax) = plt.subplots(2, 1, figsize=(5, 7), sharex=True)

    loss_ax.plot(epochs, train_loss, label="train")
    loss_ax.plot(epochs, val_loss, label="validation")
    loss_ax.set_ylabel("Loss")
    loss_ax.set_title(f"{title_prefix} — loss")
    loss_ax.legend()

    acc_ax.plot(epochs, train_accuracy, label="train")
    acc_ax.plot(epochs, val_accuracy, label="validation")
    acc_ax.set_xlabel("Epoch")
    acc_ax.set_ylabel("Accuracy")
    acc_ax.set_title(f"{title_prefix} — accuracy")
    acc_ax.legend()

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path)

    return fig
