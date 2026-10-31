"""Build a CSV manifest for an MVTec-AD category.

The manifest lists every image in the category's train/ and test/ splits,
together with its label (0 = good, 1 = defective), defect type, and the
path to the corresponding ground-truth mask (empty for good images).

Usage:
    python -m src.data.create_manifest --category screw
    python -m src.data.create_manifest --category screw --output data/manifests/screw.csv
"""

import argparse
import csv
from pathlib import Path

DEFAULT_DATA_ROOT = Path("data/mvtec_anomaly_detection")


def build_manifest(category: str, data_root: Path) -> list[dict]:
    """Collect manifest rows for a single MVTec-AD category."""
    category_dir = data_root / category
    train_dir = category_dir / "train"
    test_dir = category_dir / "test"
    ground_truth_dir = category_dir / "ground_truth"

    rows = []

    # Train split: only "good" images exist.
    train_good_dir = train_dir / "good"
    for image_path in sorted(train_good_dir.glob("*.png")):
        rows.append(
            {
                "image_path": image_path.as_posix(),
                "mask_path": "",
                "split": "train",
                "label": 0,
                "defect_type": "good",
            }
        )

    # Test split: "good" plus one subfolder per defect type.
    for defect_dir in sorted(test_dir.iterdir()):
        if not defect_dir.is_dir():
            continue
        defect_type = defect_dir.name
        is_good = defect_type == "good"

        for image_path in sorted(defect_dir.glob("*.png")):
            mask_path = ""
            if not is_good:
                mask_candidate = (
                    ground_truth_dir / defect_type / f"{image_path.stem}_mask.png"
                )
                if mask_candidate.exists():
                    mask_path = mask_candidate.as_posix()

            rows.append(
                {
                    "image_path": image_path.as_posix(),
                    "mask_path": mask_path,
                    "split": "test",
                    "label": 0 if is_good else 1,
                    "defect_type": defect_type,
                }
            )

    return rows


def write_manifest(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_path", "mask_path", "split", "label", "defect_type"]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--category", required=True, help="MVTec-AD category name, e.g. 'screw'"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root directory containing the MVTec-AD categories",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: data/manifests/<category>.csv)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output or Path("data/manifests") / f"{args.category}.csv"

    rows = build_manifest(args.category, args.data_root)
    write_manifest(rows, output_path)
    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
