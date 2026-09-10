"""Build synthetic defective training images via mask-guided copy-paste.

Generates extra training rows for the per-category defect-type classifier
(see src/models/train_defect_classifier.py), which is chronically
data-starved: MVTec's train/ split contains only `good` images, so every
real defect-type example has to come from the labeled test/ split -- as few
as 10 images per class for transistor.

Each synthetic image pastes a real defect region (cut out with its MVTec
ground-truth mask) onto a clean train/good background, so the defect_type
label is correct by construction. See src/preprocessing/copy_paste.py.
"""

import shutil
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from PIL import Image

from src.preprocessing.copy_paste import synthesize_defect


def load_good_backgrounds(manifest: pd.DataFrame) -> pd.DataFrame:
    """The category's clean train/ images, used as paste backgrounds."""
    return manifest[(manifest["split"] == "train") & (manifest["label"] == 0)].reset_index(drop=True)


def generate_synthetic_rows(
    defect_rows: pd.DataFrame,
    background_rows: pd.DataFrame,
    data_root: Path,
    output_dir: Path,
    per_class: int,
    seed: int,
    **paste_kwargs,
) -> pd.DataFrame:
    """Synthesize `per_class` images for each defect type in `defect_rows`.

    `defect_rows` must contain training-fold rows only -- sourcing a defect
    from a validation image would leak that image's defect pixels into
    training. Images/masks are written under `output_dir` and returned as
    manifest-schema rows referencing them.
    """
    if per_class <= 0:
        return defect_rows.iloc[0:0].copy()
    if background_rows.empty:
        raise ValueError("No train/good background images available for copy-paste synthesis.")

    usable = defect_rows[defect_rows["mask_path"].notna() & (defect_rows["mask_path"] != "")]
    if usable.empty:
        raise ValueError("Copy-paste synthesis requires defective rows with ground-truth masks.")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    rng = np.random.default_rng(seed)
    synthetic_rows: List[dict] = []

    for defect_type, group in usable.groupby("defect_type"):
        class_dir = output_dir / str(defect_type)
        class_dir.mkdir(parents=True, exist_ok=True)
        label = int(group.iloc[0]["label"])
        generated = 0
        attempts = 0
        # Retry budget covers sources whose mask turns out to be empty.
        while generated < per_class and attempts < per_class * 5:
            attempts += 1
            source = group.iloc[int(rng.integers(0, len(group)))]
            background = background_rows.iloc[int(rng.integers(0, len(background_rows)))]

            source_image = Image.open(data_root / source["image_path"]).convert("RGB")
            source_mask = np.asarray(
                Image.open(data_root / source["mask_path"]).convert("L")
            ) > 0
            background_image = Image.open(data_root / background["image_path"]).convert("RGB")

            result = synthesize_defect(
                background_image, source_image, source_mask, rng, **paste_kwargs
            )
            if result is None:
                continue
            composite, pasted_mask = result

            image_path = class_dir / f"{generated:04d}.png"
            mask_path = class_dir / f"{generated:04d}_mask.png"
            composite.save(image_path)
            Image.fromarray(pasted_mask.astype(np.uint8) * 255).save(mask_path)

            synthetic_rows.append(
                {
                    "image_path": str(image_path.relative_to(data_root)),
                    "mask_path": str(mask_path.relative_to(data_root)),
                    "split": "train",
                    "label": label,
                    "defect_type": defect_type,
                }
            )
            generated += 1

        if generated < per_class:
            raise RuntimeError(
                f"Only generated {generated}/{per_class} synthetic images for '{defect_type}'."
            )

    return pd.DataFrame(synthetic_rows)
