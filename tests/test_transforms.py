import pytest
from PIL import Image

from src.preprocessing.transform import get_patchcore_train_transforms


def test_patchcore_translation_preserves_tensor_shape():
    image = Image.new("RGB", (320, 240), color=(120, 100, 80))

    transformed = get_patchcore_train_transforms(224, translate_ratio=0.05)(image)

    assert transformed.shape == (3, 224, 224)


@pytest.mark.parametrize("translate_ratio", [-0.01, 0.5])
def test_patchcore_translation_rejects_invalid_ratio(translate_ratio):
    with pytest.raises(ValueError, match="translate_ratio"):
        get_patchcore_train_transforms(224, translate_ratio=translate_ratio)