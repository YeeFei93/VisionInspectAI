"""Baseline supervised good-vs-defective image classifier."""

import torch.nn as nn
from torchvision import models

SUPPORTED_ARCHITECTURES = {"resnet18", "efficientnet_b0"}


def build_baseline_model(
    architecture: str = "resnet18",
    num_classes: int = 2,
    pretrained: bool = True,
) -> nn.Module:
    """Build a baseline classifier by replacing the final layer of a
    torchvision backbone with a `num_classes`-way linear head."""
    architecture = architecture.lower()

    if architecture == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif architecture == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(
            f"Unsupported architecture '{architecture}'. "
            f"Choose one of {sorted(SUPPORTED_ARCHITECTURES)}"
        )

    return model
