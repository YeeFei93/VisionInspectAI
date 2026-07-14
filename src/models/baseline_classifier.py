"""Baseline supervised good-vs-defective image classifiers."""

import torch.nn as nn
from torchvision import models

SUPPORTED_ARCHITECTURES = {"resnet18", "efficientnet_b0", "simple_cnn"}


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
    """Build a baseline classifier. For torchvision backbones, replaces the
    final layer with a `num_classes`-way linear head. `simple_cnn` is a
    small from-scratch sequential CNN (no pretrained weights)."""
    architecture = architecture.lower()

    if architecture == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif architecture == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif architecture == "simple_cnn":
        model = SimpleCNN(num_classes=num_classes)
    else:
        raise ValueError(
            f"Unsupported architecture '{architecture}'. "
            f"Choose one of {sorted(SUPPORTED_ARCHITECTURES)}"
        )

    return model
