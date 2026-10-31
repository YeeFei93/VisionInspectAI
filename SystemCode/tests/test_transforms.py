import numpy as np
import pytest
from PIL import Image

from src.preprocessing.transform import (
    DEFAULT_DEFECT_VARIANTS,
    get_defect_augmentation_variants,
    get_defect_variant_transform,
    get_patchcore_train_transforms,
    get_train_transforms,
)


def test_patchcore_translation_preserves_tensor_shape():
    image = Image.new("RGB", (320, 240), color=(120, 100, 80))

    transformed = get_patchcore_train_transforms(224, translate_ratio=0.05)(image)

    assert transformed.shape == (3, 224, 224)


def test_train_transforms_have_no_random_rotation():
    names = [type(step).__name__ for step in get_train_transforms().transforms]

    assert "RandomRotation" not in names


@pytest.mark.parametrize(
    "category",
    [
        "screw", "bottle", "hazelnut", "carpet", "leather", "wood",
        "grid", "tile", "cable", "capsule", "metal_nut", "pill", "zipper",
    ],
)
def test_default_categories_use_all_flips_and_quarter_turns(category):
    variants = get_defect_augmentation_variants(category, "any_defect")

    assert variants == DEFAULT_DEFECT_VARIANTS
    assert {"hflip", "vflip", "rot90", "rot180", "rot270"} <= set(variants)


def test_transistor_variants_depend_on_defect_type():
    assert get_defect_augmentation_variants("transistor", "bent_lead") == ("identity", "hflip")
    assert get_defect_augmentation_variants("transistor", "misplaced") == (
        "identity",
        "hflip",
        "vflip",
        "rot180",
    )


def test_variants_are_deterministic_geometric_transforms():
    image = Image.new("RGB", (4, 4))
    image.putpixel((0, 0), (255, 0, 0))
    expected_red_position = {
        "identity": (0, 0),
        "hflip": (3, 0),
        "vflip": (0, 3),
        "rot180": (3, 3),
        "rot90": (0, 3),
        "rot270": (3, 0),
    }

    for variant, (x, y) in expected_red_position.items():
        transformed = get_defect_variant_transform(variant, 4)(image)
        assert transformed.shape == (3, 4, 4)
        assert np.argmax(transformed[0].numpy()) == y * 4 + x, variant


@pytest.mark.parametrize("translate_ratio", [-0.01, 0.5])
def test_patchcore_translation_rejects_invalid_ratio(translate_ratio):
    with pytest.raises(ValueError, match="translate_ratio"):
        get_patchcore_train_transforms(224, translate_ratio=translate_ratio)