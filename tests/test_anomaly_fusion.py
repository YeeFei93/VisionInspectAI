import numpy as np
import pytest

from src.models.anomaly_fusion import CalibrationNormalizer, fuse


def test_zscore_normalizer_centers_and_scales_calibration_values():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    normalized = CalibrationNormalizer.fit(values, "zscore")(values)
    assert normalized.mean() == pytest.approx(0.0)
    assert normalized.std() == pytest.approx(1.0)


def test_minmax_normalizer_maps_calibration_range_to_unit_interval_without_clipping():
    normalizer = CalibrationNormalizer.fit([2.0, 4.0, 6.0], "minmax")
    assert normalizer([2.0, 6.0]).tolist() == [0.0, 1.0]
    assert normalizer(10.0) == pytest.approx(2.0)  # unseen anomalies are not clipped


def test_normalizer_rejects_constant_or_empty_calibration_values():
    with pytest.raises(ValueError):
        CalibrationNormalizer.fit([1.0, 1.0], "zscore")
    with pytest.raises(ValueError):
        CalibrationNormalizer.fit([], "minmax")
    with pytest.raises(ValueError):
        CalibrationNormalizer.fit([1.0, 2.0], "bogus")


def test_fuse_is_weighted_average_and_alpha_endpoints_select_one_detector():
    a, b = np.array([1.0, 3.0]), np.array([3.0, 7.0])
    assert fuse(a, b, 0.5).tolist() == [2.0, 5.0]
    assert fuse(a, b, 1.0).tolist() == a.tolist()
    assert fuse(a, b, 0.0).tolist() == b.tolist()
    with pytest.raises(ValueError):
        fuse(a, b, 1.5)
