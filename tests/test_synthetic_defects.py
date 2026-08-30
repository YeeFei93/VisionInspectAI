"""Unit tests for src/data/generate_synthetic_defects.py.

Covers the pure, model-free functions (mask generation, Canny edge control
image, manifest writing) plus an end-to-end smoke test of
generate_synthetic_defects() with a stub diffusion pipeline injected via
pipeline_factory, so none of this requires the optional
diffusers/transformers/accelerate dependencies or downloading model
weights.
"""

import random

import numpy as np
import pandas as pd
from PIL import Image

from src.data.generate_synthetic_defects import (
    MANIFEST_COLUMNS,
    generate_synthetic_defects,
    make_canny_control_image,
    make_random_mask,
    write_synthetic_manifest,
)


def test_make_random_mask_shapes_and_size():
    for mode in ("scratch", "blob", "patch"):
        mask = make_random_mask(64, mode, random.Random(0))
        assert mask.mode == "L"
        assert mask.size == (64, 64)
        arr = np.array(mask)
        assert arr.max() == 255
        assert arr.min() == 0
        # Non-trivial region marked, but far from covering the whole image.
        assert 0 < (arr > 0).mean() < 0.5


def test_make_random_mask_deterministic_with_same_seed():
    mask_a = make_random_mask(64, "scratch", random.Random(42))
    mask_b = make_random_mask(64, "scratch", random.Random(42))
    assert np.array_equal(np.array(mask_a), np.array(mask_b))


def test_make_random_mask_rejects_unknown_mode():
    try:
        make_random_mask(64, "not-a-mode", random.Random(0))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_make_canny_control_image_matches_source_size():
    image = Image.new("RGB", (64, 64), (128, 128, 128))
    control = make_canny_control_image(image)
    assert control.size == (64, 64)
    assert control.mode == "RGB"


def test_write_synthetic_manifest_appends_across_calls(tmp_path):
    rows_1 = [{"image_path": "a.png", "mask_path": "a_mask.png", "split": "train", "label": 1, "defect_type": "x"}]
    rows_2 = [{"image_path": "b.png", "mask_path": "b_mask.png", "split": "train", "label": 1, "defect_type": "x"}]

    path = write_synthetic_manifest(rows_1, "screw", tmp_path)
    write_synthetic_manifest(rows_2, "screw", tmp_path)

    df = pd.read_csv(path)
    assert list(df.columns) == MANIFEST_COLUMNS
    assert len(df) == 2
    assert set(df["image_path"]) == {"a.png", "b.png"}


class _StubPipelineOutput:
    def __init__(self, images):
        self.images = images


class _StubPipeline:
    """Echoes the mask back as the 'generated' image so the surrounding
    plumbing (file saving, manifest rows) can be exercised without a real
    Stable Diffusion model."""

    def __call__(self, prompt, negative_prompt, image, mask_image, control_image,
                 controlnet_conditioning_scale, num_inference_steps, generator):
        return _StubPipelineOutput([mask_image.convert("RGB")])


def test_generate_synthetic_defects_end_to_end_with_stub_pipeline(tmp_path):
    data_root = tmp_path / "data"
    output_root = tmp_path / "synthetic"
    train_good_dir = data_root / "screw" / "train" / "good"
    train_good_dir.mkdir(parents=True)
    for i in range(3):
        Image.new("RGB", (32, 32), (200, 200, 200)).save(train_good_dir / f"{i:03d}.png")

    rows = generate_synthetic_defects(
        category="screw",
        defect_type="synthetic_test",
        prompt="a test defect",
        num_images=2,
        mask_mode="patch",
        image_size=32,
        controlnet_conditioning_scale=0.8,
        num_inference_steps=1,
        seed=0,
        data_root=data_root,
        output_root=output_root,
        pipeline_factory=lambda device: _StubPipeline(),
    )

    assert len(rows) == 2
    for row in rows:
        assert row["split"] == "train"
        assert row["label"] == 1
        assert row["defect_type"] == "synthetic_test"

    image_dir = output_root / "screw" / "synthetic_test"
    mask_dir = output_root / "screw" / "ground_truth" / "synthetic_test"
    assert len(list(image_dir.glob("*.png"))) == 2
    assert len(list(mask_dir.glob("*.png"))) == 2
