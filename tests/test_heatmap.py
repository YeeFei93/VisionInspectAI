"""Unit tests for src/visualization/heatmap.py normalization helpers."""

import numpy as np
import torch

from src.visualization.heatmap import normalize_map, normalize_map_threshold


def test_normalize_map_scales_to_unit_range():
    anomaly_map = torch.tensor([[0.0, 5.0], [10.0, 2.5]])
    normalized = normalize_map(anomaly_map)
    assert normalized.min() == 0.0
    assert normalized.max() == 1.0


def test_normalize_map_constant_input_returns_zeros():
    anomaly_map = torch.full((3, 3), 5.0)
    normalized = normalize_map(anomaly_map)
    assert np.allclose(normalized, 0.0)


def test_normalize_map_threshold_below_values_stay_in_cool_span():
    anomaly_map = torch.tensor([0.0, 1.0, 1.5])
    threshold = 2.0
    normalized = normalize_map_threshold(anomaly_map, threshold, vmin=0.0, vmax=3.0)
    assert (normalized <= 0.4 + 1e-8).all()


def test_normalize_map_threshold_above_values_stay_in_warm_span():
    anomaly_map = torch.tensor([2.5, 3.0])
    threshold = 2.0
    normalized = normalize_map_threshold(anomaly_map, threshold, vmin=0.0, vmax=3.0)
    assert (normalized >= 0.6 - 1e-8).all()


def test_normalize_map_threshold_is_monotonic_across_the_cliff():
    anomaly_map = torch.tensor([1.9, 2.1])
    threshold = 2.0
    normalized = normalize_map_threshold(anomaly_map, threshold, vmin=0.0, vmax=3.0)
    assert normalized[1] > normalized[0]
