import json

import pytest

from src.evaluation.summarize_patchcore_scoring import (
    SUMMARY_METRICS,
    summarize_scoring_runs,
)


def write_metrics(path, value, **settings):
    metrics = {metric: value for metric in SUMMARY_METRICS}
    metrics.update(settings)
    path.write_text(json.dumps(metrics))


def test_summarize_scoring_runs_ranks_and_computes_baseline_delta(tmp_path):
    prefix = "patchcore_resnet18_screw"
    write_metrics(tmp_path / f"{prefix}_metrics.json", 0.5)
    settings = {
        "softmax_reweighting": False,
        "reweight_num_neighbors": 9,
    }
    write_metrics(tmp_path / f"{prefix}_knn3_metrics.json", 0.7, num_neighbors=3, **settings)
    write_metrics(tmp_path / f"{prefix}_knn5_metrics.json", 0.6, num_neighbors=5, **settings)

    summary = summarize_scoring_runs(tmp_path, "screw")

    assert summary["runs"][0]["num_neighbors"] == 3
    assert summary["runs"][0]["metrics"]["auroc"]["delta_from_baseline"] == pytest.approx(0.2)


def test_summarize_scoring_runs_supports_tuned_baseline_suffix(tmp_path):
    prefix = "patchcore_resnet18_screw_cs4000_pd256_layer2"
    write_metrics(tmp_path / f"{prefix}_metrics.json", 0.5)
    write_metrics(
        tmp_path / f"{prefix}_knn3_metrics.json",
        0.6,
        num_neighbors=3,
        softmax_reweighting=False,
        reweight_num_neighbors=9,
    )

    summary = summarize_scoring_runs(
        tmp_path, "screw", baseline_suffix="_cs4000_pd256_layer2"
    )

    assert summary["baseline_name"] == prefix


def test_default_summary_excludes_scoring_runs_from_tuned_config(tmp_path):
    prefix = "patchcore_resnet18_screw"
    settings = {
        "num_neighbors": 3,
        "softmax_reweighting": False,
        "reweight_num_neighbors": 9,
    }
    write_metrics(tmp_path / f"{prefix}_metrics.json", 0.5)
    write_metrics(tmp_path / f"{prefix}_knn3_metrics.json", 0.6, **settings)
    write_metrics(
        tmp_path / f"{prefix}_cs4000_pd256_layer2_knn3_metrics.json",
        0.9,
        **settings,
    )

    summary = summarize_scoring_runs(tmp_path, "screw")

    assert [run["name"] for run in summary["runs"]] == [f"{prefix}_knn3"]