"""Baseline supervised good-vs-defective image classifiers."""

import torch.nn as nn
from torchvision import models

SUPPORTED_ARCHITECTURES = {
    "convnext_tiny",
    "efficientnet_b0",
    "resnet18",
    "simple_cnn",
    "vit_b_16",
}


class SimpleCNN(nn.Module):
    """A small sequential CNN trained from scratch (no ImageNet pretraining),
    used as a lightweight point of comparison against transfer-learning
    backbones like ResNet18/EfficientNet-B0."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def build_baseline_model(
    architecture: str = "resnet18",
    num_classes: int = 2,
    pretrained: bool = True,
) -> nn.Module:
    """Build a baseline classifier. For torchvision CNN/ViT backbones,
    replaces the final layer with a `num_classes`-way linear head.
    `simple_cnn` is a small from-scratch CNN (no pretrained weights)."""
    architecture = architecture.lower()

    if architecture == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif architecture == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif architecture == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = models.convnext_tiny(weights=weights)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    elif architecture == "vit_b_16":
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        model = models.vit_b_16(weights=weights)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    elif architecture == "simple_cnn":
        model = SimpleCNN(num_classes=num_classes)
    else:
        raise ValueError(
            f"Unsupported architecture '{architecture}'. "
            f"Choose one of {sorted(SUPPORTED_ARCHITECTURES)}"
        )

    return model


def freeze_backbone(model: nn.Module, architecture: str) -> nn.Module:
    """Freeze every parameter except the final classification head, so only
    the head is trained. Cuts trainable parameters from millions to a few
    thousand — an overfitting control for tiny datasets (e.g. per-category
    defect-type classification with ~7-20 images/class)."""
    architecture = architecture.lower()

    for param in model.parameters():
        param.requires_grad = False

    if architecture == "resnet18":
        head = model.fc
    elif architecture == "efficientnet_b0":
        head = model.classifier[1]
    elif architecture == "convnext_tiny":
        head = model.classifier[2]
    elif architecture == "vit_b_16":
        head = model.heads.head
    elif architecture == "simple_cnn":
        head = model.classifier
    else:
        raise ValueError(
            f"Unsupported architecture '{architecture}'. "
            f"Choose one of {sorted(SUPPORTED_ARCHITECTURES)}"
        )

    for param in head.parameters():
        param.requires_grad = True

    return model


def unfreeze_last_block(model: nn.Module, architecture: str) -> nn.Module:
    """Freeze everything except the classification head and the final
    backbone block/stage, then train both together. A middle ground between
    `freeze_backbone` (head only) and full fine-tuning: the most
    task-specific, least general backbone features get to adapt while early,
    more generic features stay fixed -- the two-phase strategy from the
    classic Keras "little data" fine-tuning recipe (train the head first,
    then unfreeze only the last conv block)."""
    architecture = architecture.lower()

    for param in model.parameters():
        param.requires_grad = False

    if architecture == "resnet18":
        head = model.fc
        last_block_modules = [model.layer4]
    elif architecture == "efficientnet_b0":
        head = model.classifier[1]
        last_block_modules = list(model.features.children())[-2:]
    elif architecture == "convnext_tiny":
        head = model.classifier[2]
        last_block_modules = list(model.features.children())[-2:]
    elif architecture == "vit_b_16":
        head = model.heads.head
        last_block_modules = [model.encoder.layers[-1], model.encoder.ln]
    elif architecture == "simple_cnn":
        head = model.classifier
        last_block_modules = list(model.features.children())[-4:]
    else:
        raise ValueError(
            f"Unsupported architecture '{architecture}'. "
            f"Choose one of {sorted(SUPPORTED_ARCHITECTURES)}"
        )

    for param in head.parameters():
        param.requires_grad = True
    for module in last_block_modules:
        for param in module.parameters():
            param.requires_grad = True

    return model


FREEZE_MODES = {"none", "all", "last_block"}


def apply_freeze_mode(model: nn.Module, architecture: str, freeze_mode: str) -> nn.Module:
    """Dispatch to an overfitting-control strategy for tiny per-category
    datasets: `none` fine-tunes every parameter, `all` trains only the head
    (`freeze_backbone`), and `last_block` trains the head plus the final
    backbone block/stage (`unfreeze_last_block`)."""
    if freeze_mode == "none":
        return model
    if freeze_mode == "all":
        return freeze_backbone(model, architecture)
    if freeze_mode == "last_block":
        return unfreeze_last_block(model, architecture)
    raise ValueError(f"Unsupported freeze_mode '{freeze_mode}'. Choose one of {sorted(FREEZE_MODES)}")
