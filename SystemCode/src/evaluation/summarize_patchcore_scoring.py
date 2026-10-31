"""Summarize PatchCore k-NN and softmax-reweighting experiments."""

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_METRICS = ("auroc", "pixel_auroc", "mean_iou", "mean_dice")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", required=True)
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--baseline-suffix", default="")
    parser.add_argument("--metrics-dir", type=Path, default=Path("outputs/metrics"))
    return parser.parse_args()


def summarize_scoring_runs(
    metrics_dir: Path,
    category: str,
    backbone: str = "resnet18",
    baseline_suffix: str = "",
) -> dict:
    prefix = f"patchcore_{backbone}_{category}"
    baseline_name = f"{prefix}{baseline_suffix}"
    baseline_path = metrics_dir / f"{baseline_name}_metrics.json"
    if not baseline_path.exists():
        raise FileNotFoundError(f"Missing scoring baseline: {baseline_path}")
    baseline = json.loads(baseline_path.read_text())

    candidates = []
    for path in sorted(metrics_dir.glob(f"{baseline_name}_*_metrics.json")):
        candidate_suffix = path.stem.removeprefix(f"{baseline_name}_")
        if not candidate_suffix.startswith(("knn", "rw")):
            continue
        metrics = json.loads(path.read_text())
        candidates.append(
            {
                "name": path.stem.removesuffix("_metrics"),
                "num_neighbors": int(metrics["num_neighbors"]),
                "softmax_reweighting": bool(metrics["softmax_reweighting"]),
                "reweight_num_neighbors": int(metrics["reweight_num_neighbors"]),
                "metrics": {
                    metric: {
                        "value": float(metrics[metric]),
                        "delta_from_baseline": float(metrics[metric])
                        - float(baseline[metric]),
                    }
                    for metric in SUMMARY_METRICS
                },
            }
        )
    if not candidates:
        raise FileNotFoundError(f"No scoring experiments found for {baseline_name}")
    candidates.sort(key=lambda run: run["metrics"]["auroc"]["value"], reverse=True)
    return {
        "category": category,
        "backbone": backbone,
        "baseline_name": baseline_name,
        "baseline": {metric: float(baseline[metric]) for metric in SUMMARY_METRICS},
        "runs": candidates,
    }


def main() -> None:
    args = parse_args()
    metrics_dir = PROJECT_ROOT / args.metrics_dir
    summary = summarize_scoring_runs(
        metrics_dir, args.category, args.backbone, args.baseline_suffix
    )
    output_path = metrics_dir / f"{summary['baseline_name']}_scoring_summary.json"
    output_path.write_text(json.dumps(summary, indent=2))
    for run in summary["runs"]:
        image = run["metrics"]["auroc"]
        pixel = run["metrics"]["pixel_auroc"]
        print(
            f"{run['name']}: image AUROC {image['value']:.4f} "
            f"(delta {image['delta_from_baseline']:+.4f}); "
            f"pixel AUROC {pixel['value']:.4f} "
            f"(delta {pixel['delta_from_baseline']:+.4f})"
        )
    print(f"Saved summary to {output_path}")


if __name__ == "__main__":
    main()