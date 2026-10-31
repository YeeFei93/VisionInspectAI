"""Bundle the artifacts the Streamlit demo needs into release-sized zip files.

Instead of hardcoding file names, the script instantiates InspectionPipeline,
loads every category and records which checkpoint/metrics files are actually
opened, so experiment-only artifacts (ViT baselines, autoencoders, ...) are skipped.

Usage:
    python scripts/package_models.py                  # writes dist/models/*.zip
    python scripts/package_models.py --max-part-mb 1500

Then upload dist/models/*.zip to a GitHub Release (each asset must be < 2 GiB).
"""
import argparse
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

ARTIFACT_DIRS = (
    PROJECT_ROOT / "models" / "checkpoints",
    PROJECT_ROOT / "outputs" / "metrics",
)
ZIP_PREFIX = "visioninspect-models"


def collect_required_files() -> list[Path]:
    from src.inference.inspection_pipeline import (
        DEFAULT_CATEGORY_CONFIGS,
        InspectionPipeline,
    )

    opened: set[Path] = set()

    def audit_hook(event: str, args: tuple) -> None:
        if event == "open" and isinstance(args[0], str):
            path = Path(args[0]).resolve()
            if any(path.is_relative_to(directory) for directory in ARTIFACT_DIRS):
                opened.add(path)

    sys.addaudithook(audit_hook)
    pipeline = InspectionPipeline(project_root=PROJECT_ROOT)
    _ = pipeline.known_categories
    for category in DEFAULT_CATEGORY_CONFIGS:
        config = pipeline._load_config(category)
        pipeline._load_detector(category, config)
        pipeline._load_metrics(category, config)
        pipeline._load_defect_classifier(category, config)  # None is fine (optional)
    return sorted(opened)


def split_into_parts(files: list[Path], max_part_bytes: int) -> list[list[Path]]:
    parts: list[tuple[int, list[Path]]] = []
    for path in sorted(files, key=lambda p: p.stat().st_size, reverse=True):
        size = path.stat().st_size
        for index, (used, members) in enumerate(parts):
            if used + size <= max_part_bytes:
                parts[index] = (used + size, members + [path])
                break
        else:
            parts.append((size, [path]))
    return [members for _, members in parts]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "dist" / "models")
    parser.add_argument("--max-part-mb", type=int, default=1500)
    args = parser.parse_args()

    files = collect_required_files()
    total = sum(path.stat().st_size for path in files)
    print(f"{len(files)} required files, {total / 1e6:.0f} MB total")

    parts = split_into_parts(files, args.max_part_mb * 1_000_000)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for old_zip in args.output_dir.glob(f"{ZIP_PREFIX}-*.zip"):
        old_zip.unlink()

    for number, members in enumerate(parts, start=1):
        zip_path = args.output_dir / f"{ZIP_PREFIX}-{number}of{len(parts)}.zip"
        # Weights are float tensors that barely compress, so store them as-is.
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as archive:
            for path in sorted(members):
                archive.write(path, path.relative_to(PROJECT_ROOT).as_posix())
        print(f"{zip_path.name}: {len(members)} files, {zip_path.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
