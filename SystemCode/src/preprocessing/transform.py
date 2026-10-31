"""Image transforms for supervised classifiers."""

from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

FLIP_LR = Image.Transpose.FLIP_LEFT_RIGHT
FLIP_TB = Image.Transpose.FLIP_TOP_BOTTOM
_VARIANT_OPS = {
    "identity": (),
    "hflip": (FLIP_LR,),
    "vflip": (FLIP_TB,),
    "rot90": (Image.Transpose.ROTATE_90,),
    "rot180": (Image.Transpose.ROTATE_180,),
    "rot270": (Image.Transpose.ROTATE_270,),
}
DEFAULT_DEFECT_VARIANTS = (
    "identity", "hflip", "vflip", "rot90", "rot180", "rot270",
)


def get_defect_augmentation_variants(category: str, defect_type: str) -> tuple:
    """Fixed (non-random) variants generated for one training image."""
    if category == "transistor":
        if defect_type == "misplaced":
            return ("identity", "hflip", "vflip", "rot180")
        return ("identity", "hflip")
    return DEFAULT_DEFECT_VARIANTS


class _ApplyVariant:
    def __init__(self, variant: str):
        self.operations = _VARIANT_OPS[variant]

    def __call__(self, image):
        for operation in self.operations:
            image = image.transpose(operation)
        return image


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def get_defect_variant_transform(
    variant: str, image_size: int = 224
) -> transforms.Compose:
    # Resize to a square first so quarter turns neither crop nor pad.
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            _ApplyVariant(variant),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def get_val_transforms(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def get_autoencoder_transforms(image_size: int = 224) -> transforms.Compose:
    """Raw [0, 1] pixels with no ImageNet normalization: the autoencoder
    reconstructs the image itself (sigmoid output), so the input and the
    reconstruction target must live in the same pixel space."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )


def get_patchcore_train_transforms(
    image_size: int = 224, translate_ratio: float = 0.0
) -> transforms.Compose:
    if not 0.0 <= translate_ratio < 0.5:
        raise ValueError("translate_ratio must be in [0.0, 0.5)")
    if translate_ratio == 0.0:
        return get_val_transforms(image_size)

    padding = max(1, round(image_size * translate_ratio))
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.Pad(padding, padding_mode="reflect"),
            transforms.RandomCrop((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
