"""Summarize one-factor PatchCore tuning runs against the default baseline.

Usage:
    python -m src.evaluation.summarize_patchcore_tuning --category screw
"""

import argparse
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_METRICS = ("auroc", "pixel_auroc", "mean_iou", "mean_dice")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", required=True)
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--metrics-dir", type=Path, default=Path("outputs/metrics"))
    return parser.parse_args()


def summarize_tuning_runs(metrics_dir: Path, category: str, backbone: str = "resnet18") -> dict:
    prefix = f"patchcore_{backbone}_{category}"
    baseline_path = metrics_dir / f"{prefix}_metrics.json"
    if not baseline_path.exists():
        raise FileNotFoundError(f"Missing PatchCore baseline metrics: {baseline_path}")

    baseline = json.loads(baseline_path.read_text())
    candidate_paths = sorted(
        path
        for path in metrics_dir.glob(f"{prefix}_*_metrics.json")
        if any(token in path.stem for token in ("_cs", "_pd", "_layer"))
    )
    if not candidate_paths:
        raise FileNotFoundError(f"No tuning metrics found for {category} in {metrics_dir}")

    runs = []
    for path in candidate_paths:
        metrics = json.loads(path.read_text())
        runs.append(
            {
                "name": path.stem.removesuffix("_metrics"),
                "seed": int(metrics.get("seed", 42)),
                "max_coreset_size": metrics["max_coreset_size"],
                "memory_bank_size": metrics["memory_bank_size"],
                "projection_dim": metrics["projection_dim"],
                "layers": metrics["layers"],
                "metrics": {
                    metric: {
                        "value": float(metrics[metric]),
                        "delta_from_baseline": float(metrics[metric]) - float(baseline[metric]),
                    }
                    for metric in SUMMARY_METRICS
                },
            }
        )

    runs.sort(key=lambda run: run["metrics"]["auroc"]["value"], reverse=True)
    grouped_runs = {}
    for run in runs:
        key = (
            run["max_coreset_size"],
            run["projection_dim"],
            tuple(run["layers"]),
        )
        grouped_runs.setdefault(key, []).append(run)

    configurations = []
    for (coreset_size, projection_dim, layers), matching_runs in grouped_runs.items():
        metric_summary = {}
        for metric in SUMMARY_METRICS:
            values = np.asarray(
                [run["metrics"][metric]["value"] for run in matching_runs], dtype=float
            )
            mean = float(values.mean())
            metric_summary[metric] = {
                "mean": mean,
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "delta_from_baseline": mean - float(baseline[metric]),
            }
        configurations.append(
            {
                "max_coreset_size": coreset_size,
                "projection_dim": projection_dim,
                "layers": list(layers),
                "seeds": sorted(run["seed"] for run in matching_runs),
                "metrics": metric_summary,
            }
        )
    configurations.sort(
        key=lambda config: config["metrics"]["auroc"]["mean"], reverse=True
    )

    return {
        "category": category,
        "backbone": backbone,
        "baseline": {metric: float(baseline[metric]) for metric in SUMMARY_METRICS},
        "ranking_metric": "auroc",
        "configurations": configurations,
        "runs": runs,
    }


def main() -> None:
    args = parse_args()
    metrics_dir = PROJECT_ROOT / args.metrics_dir
    summary = summarize_tuning_runs(metrics_dir, args.category, args.backbone)
    output_path = metrics_dir / f"patchcore_{args.backbone}_{args.category}_tuning_summary.json"
    output_path.write_text(json.dumps(summary, indent=2))

    for config in summary["configurations"]:
        image = config["metrics"]["auroc"]
        pixel = config["metrics"]["pixel_auroc"]
        print(
            f"coreset={config['max_coreset_size']}, projection_dim={config['projection_dim']}, "
            f"layers={'+'.join(config['layers'])}, seeds={config['seeds']}: "
            f"image AUROC {image['mean']:.4f} +/- {image['std']:.4f} "
            f"(delta {image['delta_from_baseline']:+.4f}); "
            f"pixel AUROC {pixel['mean']:.4f} +/- {pixel['std']:.4f} "
            f"(delta {pixel['delta_from_baseline']:+.4f})"
        )
    print(f"Saved summary to {output_path}")


if __name__ == "__main__":
    main()