"""Summarize a seeded PatchCore coreset-projection benchmark.

Usage:
    python -m src.evaluation.summarize_projection_benchmark \
        --categories screw bottle hazelnut carpet leather \
        --seeds 41 42 43
"""

import argparse
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_METRICS = ("auroc", "pixel_auroc", "accuracy", "f1", "mean_iou", "mean_dice")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--categories", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--projection-method", choices=("pca",), default="pca")
    parser.add_argument("--baseline-seed", type=int, default=42)
    parser.add_argument("--metrics-dir", type=Path, default=Path("outputs/metrics"))
    return parser.parse_args()


def load_metrics(path: Path, description: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description} metrics: {path}")
    return json.loads(path.read_text())


def summarize_projection_runs(
    metrics_dir: Path,
    categories,
    seeds,
    backbone: str = "resnet18",
    projection_method: str = "pca",
    baseline_seed: int = 42,
) -> dict:
    category_summaries = {}
    for category in categories:
        legacy_baseline_path = metrics_dir / f"patchcore_{backbone}_{category}_metrics.json"
        seeded_baseline_path = metrics_dir / (
            f"patchcore_{backbone}_{category}_seed{baseline_seed}_metrics.json"
        )
        baseline_path = (
            legacy_baseline_path if legacy_baseline_path.exists() else seeded_baseline_path
        )
        baseline = load_metrics(baseline_path, f"random-projection baseline for {category}")
        baseline_projection = baseline.get("projection_method", "random")
        recorded_baseline_seed = int(baseline.get("seed", baseline_seed))
        if baseline_projection != "random":
            raise ValueError(f"Baseline metrics must use random projection: {baseline_path}")
        if recorded_baseline_seed != baseline_seed:
            raise ValueError(
                f"Baseline seed mismatch in {baseline_path}: "
                f"expected {baseline_seed}, found {recorded_baseline_seed}"
            )
        runs = []
        for seed in seeds:
            run_path = metrics_dir / (
                f"patchcore_{backbone}_{category}_{projection_method}proj_seed{seed}_metrics.json"
            )
            runs.append(load_metrics(run_path, f"{projection_method}-projection run for {category}, seed {seed}"))

        metric_summary = {}
        for metric in SUMMARY_METRICS:
            values = np.asarray([run[metric] for run in runs], dtype=float)
            baseline_value = float(baseline[metric])
            mean = float(values.mean())
            metric_summary[metric] = {
                "baseline": baseline_value,
                "mean": mean,
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "delta_from_baseline": mean - baseline_value,
            }

        category_summaries[category] = {
            "metrics": metric_summary,
            "runs": [
                {"seed": seed, **{metric: run[metric] for metric in SUMMARY_METRICS}}
                for seed, run in zip(seeds, runs)
            ],
        }

    return {
        "backbone": backbone,
        "projection_method": projection_method,
        "seeds": list(seeds),
        "baseline_projection_method": "random",
        "baseline_seed": baseline_seed,
        "categories": category_summaries,
    }


def main() -> None:
    args = parse_args()
    metrics_dir = PROJECT_ROOT / args.metrics_dir
    summary = summarize_projection_runs(
        metrics_dir,
        args.categories,
        args.seeds,
        backbone=args.backbone,
        projection_method=args.projection_method,
        baseline_seed=args.baseline_seed,
    )
    output_path = metrics_dir / (
        f"patchcore_{args.backbone}_{args.projection_method}proj_multiseed_summary.json"
    )
    output_path.write_text(json.dumps(summary, indent=2))

    for category, category_summary in summary["categories"].items():
        auroc = category_summary["metrics"]["auroc"]
        pixel_auroc = category_summary["metrics"]["pixel_auroc"]
        print(
            f"{category}: image AUROC {auroc['mean']:.4f} ± {auroc['std']:.4f} "
            f"(Δ {auroc['delta_from_baseline']:+.4f}); "
            f"pixel AUROC {pixel_auroc['mean']:.4f} ± {pixel_auroc['std']:.4f} "
            f"(Δ {pixel_auroc['delta_from_baseline']:+.4f})"
        )
    print(f"Saved summary to {output_path}")


if __name__ == "__main__":
    main()