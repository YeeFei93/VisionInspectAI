import json

import pytest

from src.evaluation.summarize_patchcore_tuning import (
    SUMMARY_METRICS,
    summarize_tuning_runs,
)


def write_metrics(path, value, **settings):
    metrics = {metric: value for metric in SUMMARY_METRICS}
    metrics.update(settings)
    path.write_text(json.dumps(metrics))


def test_summarize_tuning_runs_ranks_by_image_auroc_and_computes_deltas(tmp_path):
    write_metrics(tmp_path / "patchcore_resnet18_screw_metrics.json", 0.5)
    settings = {
        "max_coreset_size": 1000,
        "memory_bank_size": 1000,
        "projection_dim": 128,
        "layers": ["layer2", "layer3"],
    }
    write_metrics(tmp_path / "patchcore_resnet18_screw_cs1000_metrics.json", 0.6, **settings)
    write_metrics(tmp_path / "patchcore_resnet18_screw_pd64_metrics.json", 0.7, **settings)

    summary = summarize_tuning_runs(tmp_path, "screw")

    assert [run["name"] for run in summary["runs"]] == [
        "patchcore_resnet18_screw_pd64",
        "patchcore_resnet18_screw_cs1000",
    ]
    assert summary["runs"][0]["metrics"]["auroc"]["delta_from_baseline"] == pytest.approx(0.2)


def test_summarize_tuning_runs_groups_matching_settings_across_seeds(tmp_path):
    write_metrics(tmp_path / "patchcore_resnet18_screw_metrics.json", 0.5)
    settings = {
        "max_coreset_size": 4000,
        "memory_bank_size": 4000,
        "projection_dim": 256,
        "layers": ["layer2"],
    }
    for seed, value in ((41, 0.6), (42, 0.8)):
        write_metrics(
            tmp_path / f"patchcore_resnet18_screw_seed{seed}_cs4000_pd256_layer2_metrics.json",
            value,
            seed=seed,
            **settings,
        )

    summary = summarize_tuning_runs(tmp_path, "screw")
    config = summary["configurations"][0]

    assert config["seeds"] == [41, 42]
    assert config["metrics"]["auroc"]["mean"] == pytest.approx(0.7)
    assert config["metrics"]["auroc"]["std"] == pytest.approx(0.1414213562)


def test_summarize_tuning_runs_requires_candidates(tmp_path):
    write_metrics(tmp_path / "patchcore_resnet18_screw_metrics.json", 0.5)

    with pytest.raises(FileNotFoundError, match="No tuning metrics"):
        summarize_tuning_runs(tmp_path, "screw")