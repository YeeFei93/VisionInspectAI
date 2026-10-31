import json

import pytest

from src.evaluation.summarize_projection_benchmark import (
    SUMMARY_METRICS,
    summarize_projection_runs,
)


def test_summarize_projection_runs_computes_mean_std_and_delta(tmp_path):
    baseline = {metric: 0.5 for metric in SUMMARY_METRICS}
    (tmp_path / "patchcore_resnet18_screw_metrics.json").write_text(json.dumps(baseline))

    for seed, value in ((41, 0.6), (42, 0.8)):
        metrics = {metric: value for metric in SUMMARY_METRICS}
        path = tmp_path / f"patchcore_resnet18_screw_pcaproj_seed{seed}_metrics.json"
        path.write_text(json.dumps(metrics))

    summary = summarize_projection_runs(tmp_path, ["screw"], [41, 42])
    auroc = summary["categories"]["screw"]["metrics"]["auroc"]

    assert auroc["baseline"] == 0.5
    assert auroc["mean"] == pytest.approx(0.7)
    assert auroc["std"] == pytest.approx(0.1414213562)
    assert auroc["delta_from_baseline"] == pytest.approx(0.2)


def test_summarize_projection_runs_accepts_seeded_random_baseline(tmp_path):
    baseline = {
        **{metric: 0.5 for metric in SUMMARY_METRICS},
        "projection_method": "random",
        "seed": 42,
    }
    baseline_path = tmp_path / "patchcore_resnet18_screw_seed42_metrics.json"
    baseline_path.write_text(json.dumps(baseline))
    run = {metric: 0.6 for metric in SUMMARY_METRICS}
    run_path = tmp_path / "patchcore_resnet18_screw_pcaproj_seed41_metrics.json"
    run_path.write_text(json.dumps(run))

    summary = summarize_projection_runs(tmp_path, ["screw"], [41])

    assert summary["categories"]["screw"]["metrics"]["auroc"]["baseline"] == 0.5


def test_summarize_projection_runs_rejects_non_random_baseline(tmp_path):
    baseline = {
        **{metric: 0.5 for metric in SUMMARY_METRICS},
        "projection_method": "pca",
        "seed": 42,
    }
    (tmp_path / "patchcore_resnet18_screw_metrics.json").write_text(json.dumps(baseline))

    with pytest.raises(ValueError, match="must use random projection"):
        summarize_projection_runs(tmp_path, ["screw"], [41])