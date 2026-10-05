import pytest
from PIL import Image

from src.preprocessing import transform as image_transforms
from src.preprocessing.transform import (
    get_defect_classifier_train_transforms,
    get_patchcore_train_transforms,
)


def test_patchcore_translation_preserves_tensor_shape():
    image = Image.new("RGB", (320, 240), color=(120, 100, 80))

    transformed = get_patchcore_train_transforms(224, translate_ratio=0.05)(image)

    assert transformed.shape == (3, 224, 224)


def test_defect_classifier_augmentation_uses_quarter_turns(monkeypatch):
    selected = {}

    def choose_angle(angles):
        selected["angles"] = angles
        return angles[-1]

    monkeypatch.setattr(image_transforms.random, "choice", choose_angle)
    image = Image.new("RGB", (8, 8), color=(20, 40, 60))

    rotated = image_transforms._RandomQuarterTurn()(image)
    transformed = get_defect_classifier_train_transforms(32)(image)

    assert selected["angles"] == (0, 90, 180, 270)
    assert rotated.size == image.size
    assert transformed.shape == (3, 32, 32)


@pytest.mark.parametrize("translate_ratio", [-0.01, 0.5])
def test_patchcore_translation_rejects_invalid_ratio(translate_ratio):
    with pytest.raises(ValueError, match="translate_ratio"):
        get_patchcore_train_transforms(224, translate_ratio=translate_ratio)