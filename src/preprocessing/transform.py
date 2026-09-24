"""Image transforms for the baseline supervised good-vs-defective classifier."""

from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
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
