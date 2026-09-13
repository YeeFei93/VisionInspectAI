"""Tests for confidence-based inspection gating."""

from src.inference.inspection_pipeline import should_make_prediction


def test_low_confidence_blocks_prediction():
    assert should_make_prediction(0.59) is False


def test_threshold_confidence_allows_prediction():
    assert should_make_prediction(0.60) is True


def test_high_confidence_allows_prediction():
    assert should_make_prediction(0.95) is True
