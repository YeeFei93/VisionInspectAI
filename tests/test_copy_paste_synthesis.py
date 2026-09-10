import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.data.synthetic_defects import generate_synthetic_rows, load_good_backgrounds
from src.preprocessing.copy_paste import extract_defect_patch, paste_defect, synthesize_defect


def _solid_image(size, color):
    return Image.new("RGB", size, color)


def test_extract_defect_patch_crops_to_mask_bounding_box():
    image = _solid_image((10, 10), (0, 0, 0))
    mask = np.zeros((10, 10), dtype=bool)
    mask[3:6, 2:8] = True

    patch, patch_mask = extract_defect_patch(image, mask)

    assert patch.size == (6, 3)
    assert patch_mask.shape == (3, 6)
    assert patch_mask.all()


def test_extract_defect_patch_returns_none_for_empty_mask():
    image = _solid_image((10, 10), (0, 0, 0))

    assert extract_defect_patch(image, np.zeros((10, 10), dtype=bool)) is None


def test_extract_defect_patch_resizes_mismatched_mask():
    image = _solid_image((20, 20), (0, 0, 0))
    mask = np.zeros((10, 10), dtype=bool)
    mask[0:5, 0:5] = True

    patch, _patch_mask = extract_defect_patch(image, mask)

    assert patch.size == (10, 10)


def test_paste_defect_keeps_background_size_and_marks_pasted_region():
    background = _solid_image((32, 32), (0, 0, 0))
    patch = _solid_image((8, 8), (255, 0, 0))
    patch_mask = np.ones((8, 8), dtype=bool)
    rng = np.random.default_rng(0)

    composite, pasted_mask = paste_defect(
        background, patch, patch_mask, rng, scale_range=(1.0, 1.0), rotation_degrees=0, feather_radius=0
    )

    assert composite.size == background.size
    assert pasted_mask.shape == (32, 32)
    assert pasted_mask.sum() == 64
    # The pasted pixels must actually change the background.
    assert np.asarray(composite)[pasted_mask].max() > 0


def test_paste_defect_shrinks_patch_larger_than_background():
    background = _solid_image((16, 16), (0, 0, 0))
    patch = _solid_image((64, 64), (255, 0, 0))
    patch_mask = np.ones((64, 64), dtype=bool)

    composite, pasted_mask = paste_defect(
        background,
        patch,
        patch_mask,
        np.random.default_rng(0),
        scale_range=(1.0, 1.0),
        rotation_degrees=0,
        feather_radius=0,
    )

    assert composite.size == (16, 16)
    assert pasted_mask.shape == (16, 16)


def test_paste_defect_rejects_invalid_scale_range():
    with pytest.raises(ValueError, match="scale_range"):
        paste_defect(
            _solid_image((8, 8), (0, 0, 0)),
            _solid_image((4, 4), (1, 1, 1)),
            np.ones((4, 4), dtype=bool),
            np.random.default_rng(0),
            scale_range=(1.5, 0.5),
        )


def test_paste_defect_anchors_near_requested_location():
    background = _solid_image((64, 64), (0, 0, 0))
    patch = _solid_image((8, 8), (255, 0, 0))
    patch_mask = np.ones((8, 8), dtype=bool)

    _composite, pasted_mask = paste_defect(
        background,
        patch,
        patch_mask,
        np.random.default_rng(0),
        scale_range=(1.0, 1.0),
        rotation_degrees=0,
        feather_radius=0,
        anchor=(50.0, 50.0),
        jitter_ratio=0.0,
    )

    rows, columns = np.nonzero(pasted_mask)
    assert abs(columns.mean() - 50.0) <= 1
    assert abs(rows.mean() - 50.0) <= 1


def test_synthesize_defect_preserves_source_location_by_default():
    background = _solid_image((64, 64), (0, 0, 0))
    source = _solid_image((64, 64), (255, 0, 0))
    source_mask = np.zeros((64, 64), dtype=bool)
    source_mask[40:48, 40:48] = True

    _composite, pasted_mask = synthesize_defect(
        background,
        source,
        source_mask,
        np.random.default_rng(0),
        scale_range=(1.0, 1.0),
        rotation_degrees=0,
        feather_radius=0,
        jitter_ratio=0.0,
    )

    rows, columns = np.nonzero(pasted_mask)
    # Source defect is centred at (43.5, 43.5); the paste must land there too.
    assert abs(columns.mean() - 43.5) <= 1
    assert abs(rows.mean() - 43.5) <= 1


def test_synthesize_defect_returns_none_when_source_mask_empty():
    result = synthesize_defect(
        _solid_image((16, 16), (0, 0, 0)),
        _solid_image((16, 16), (255, 0, 0)),
        np.zeros((16, 16), dtype=bool),
        np.random.default_rng(0),
    )

    assert result is None


def test_load_good_backgrounds_selects_train_good_rows_only():
    manifest = pd.DataFrame(
        {
            "image_path": ["a.png", "b.png", "c.png"],
            "split": ["train", "test", "test"],
            "label": [0, 0, 1],
        }
    )

    backgrounds = load_good_backgrounds(manifest)

    assert list(backgrounds["image_path"]) == ["a.png"]


def _write_dataset(data_root):
    (data_root / "imgs").mkdir(parents=True)
    _solid_image((16, 16), (10, 10, 10)).save(data_root / "imgs" / "good.png")
    _solid_image((16, 16), (200, 0, 0)).save(data_root / "imgs" / "defect.png")
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:8, 4:8] = 255
    Image.fromarray(mask).save(data_root / "imgs" / "defect_mask.png")


def test_generate_synthetic_rows_writes_images_and_manifest_rows(tmp_path):
    _write_dataset(tmp_path)
    defect_rows = pd.DataFrame(
        {
            "image_path": ["imgs/defect.png"],
            "mask_path": ["imgs/defect_mask.png"],
            "split": ["test"],
            "label": [0],
            "defect_type": ["scratch"],
        }
    )
    backgrounds = pd.DataFrame({"image_path": ["imgs/good.png"], "split": ["train"], "label": [0]})

    rows = generate_synthetic_rows(
        defect_rows, backgrounds, tmp_path, tmp_path / "synthetic", per_class=3, seed=42
    )

    assert len(rows) == 3
    assert set(rows["defect_type"]) == {"scratch"}
    assert (rows["split"] == "train").all()
    for _, row in rows.iterrows():
        assert (tmp_path / row["image_path"]).exists()
        assert (tmp_path / row["mask_path"]).exists()


def test_generate_synthetic_rows_disabled_returns_empty_frame(tmp_path):
    defect_rows = pd.DataFrame(
        {"image_path": ["a.png"], "mask_path": ["m.png"], "label": [0], "defect_type": ["x"]}
    )

    rows = generate_synthetic_rows(
        defect_rows, pd.DataFrame(), tmp_path, tmp_path / "synthetic", per_class=0, seed=42
    )

    assert rows.empty


def test_generate_synthetic_rows_requires_masks(tmp_path):
    defect_rows = pd.DataFrame(
        {"image_path": ["a.png"], "mask_path": [""], "label": [0], "defect_type": ["x"]}
    )
    backgrounds = pd.DataFrame({"image_path": ["good.png"]})

    with pytest.raises(ValueError, match="ground-truth masks"):
        generate_synthetic_rows(
            defect_rows, backgrounds, tmp_path, tmp_path / "synthetic", per_class=1, seed=42
        )
