"""PyTorch Dataset built from a manifest CSV (see src/data/create_manifest.py).

The MVTec-AD train/ folder only contains "good" images, so a supervised
good-vs-defective classifier can't be trained from it directly. Instead we
carve a train/validation split out of the labeled test/ rows, which contain
both "good" and every defect type.
"""

from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from src.preprocessing.defect_crop import crop_to_defect


class ManifestImageDataset(Dataset):
    """Loads (image, label) pairs described by a manifest DataFrame."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        data_root: Path,
        transform: Optional[Callable] = None,
        focus_on_mask: bool = False,
        crop_padding_ratio: float = 0.25,
        min_crop_fraction: float = 0.25,
    ):
        self.manifest = manifest.reset_index(drop=True)
        self.data_root = Path(data_root)
        self.transform = transform
        self.focus_on_mask = focus_on_mask
        self.crop_padding_ratio = crop_padding_ratio
        self.min_crop_fraction = min_crop_fraction

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int):
        row = self.manifest.iloc[idx]
        image = Image.open(self.data_root / row["image_path"]).convert("RGB")
        mask_path = row.get("mask_path", "")
        focus_mask = row.get("focus_mask", None)
        if self.focus_on_mask and isinstance(focus_mask, np.ndarray):
            image = crop_to_defect(
                image,
                focus_mask,
                padding_ratio=self.crop_padding_ratio,
                min_crop_fraction=self.min_crop_fraction,
            )
        elif self.focus_on_mask and isinstance(mask_path, str) and mask_path:
            mask = np.asarray(Image.open(self.data_root / mask_path).convert("L")) > 0
            image = crop_to_defect(
                image,
                mask,
                padding_ratio=self.crop_padding_ratio,
                min_crop_fraction=self.min_crop_fraction,
            )
        if self.transform is not None:
            image = self.transform(image)
        label = int(row["label"])
        return image, label


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    return pd.read_csv(manifest_path)


def make_train_val_split(
    manifest: pd.DataFrame, val_split: float = 0.3, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/val split built from the labeled test/ rows.

    Returns:
        (train_subset, val_subset) DataFrames, both sourced from split=="test"
        rows so every row has a ground-truth good/defective label.
    """
    test_rows = manifest[manifest["split"] == "test"].reset_index(drop=True)
    train_subset, val_subset = train_test_split(
        test_rows,
        test_size=val_split,
        random_state=seed,
        stratify=test_rows["label"],
    )
    return train_subset.reset_index(drop=True), val_subset.reset_index(drop=True)
