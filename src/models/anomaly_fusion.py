"""Score/heatmap fusion for two unsupervised detectors (PatchCore +
autoencoder).

PatchCore scores (feature-space distances) and autoencoder scores
(reconstruction error) live on unrelated scales, so each is first put on a
common scale using statistics of *held-out normal calibration images only*
-- never the labelled test set -- and then combined with a fixed weight:

    fused = alpha * norm(PatchCore) + (1 - alpha) * norm(Autoencoder)

The same rule is applied to image scores and to pixel heatmaps. Test-time
values are intentionally not clipped, so a defect that is far outside the
normal range keeps contributing proportionally to the fused score.
"""

from dataclasses import dataclass

import numpy as np

NORMALIZATION_METHODS = ("zscore", "minmax")


@dataclass(frozen=True)
class CalibrationNormalizer:
    """Affine map `(x - shift) / scale` fitted on normal-only values."""

    shift: float
    scale: float

    @classmethod
    def fit(cls, normal_values, method: str = "zscore") -> "CalibrationNormalizer":
        values = np.asarray(normal_values, dtype=np.float64).ravel()
        if values.size == 0:
            raise ValueError("Cannot fit a normalizer on zero calibration values")
        if method == "zscore":
            shift, scale = values.mean(), values.std()
        elif method == "minmax":
            shift, scale = values.min(), values.max() - values.min()
        else:
            raise ValueError(f"method must be one of {NORMALIZATION_METHODS}")
        if scale <= 0:
            raise ValueError("Calibration values are constant; cannot normalize")
        return cls(shift=float(shift), scale=float(scale))

    def __call__(self, values):
        return (np.asarray(values, dtype=np.float64) - self.shift) / self.scale


def fuse(primary, secondary, alpha: float = 0.5):
    """Weighted average `alpha * primary + (1 - alpha) * secondary` of two
    already-normalized scores or anomaly maps."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    return alpha * np.asarray(primary) + (1.0 - alpha) * np.asarray(secondary)
