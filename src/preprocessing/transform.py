"""Image transforms for the baseline supervised good-vs-defective classifier."""

from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """Augmentation for the supervised classifiers (baseline good/defective,
    category, defect-type). Beyond flip+rotation, adds translate/scale/shear
    (folded into one RandomAffine call) so the model never sees the same
    crop/zoom/skew twice -- mirrors the classic Keras "little data" recipe's
    width/height shift, zoom, and shear ranges, which flip+rotation alone
    don't cover."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.85, 1.15), shear=10),
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


def get_patchcore_bank_augmentation_transforms(image_size: int = 224) -> transforms.Compose:
    """Mild augmentation for PatchCore memory-bank fitting only (never for
    scoring/inference). Expands the normal-patch manifold the coreset draws
    from with small brightness/contrast, rotation, translation, and scale
    perturbations -- deliberately conservative (no flips, no large
    rotations) since aggressive augmentation risks teaching the memory bank
    that genuine defect-like changes are normal."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.RandomAffine(degrees=5, translate=(0.02, 0.02), scale=(0.98, 1.02)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
