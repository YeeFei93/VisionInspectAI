"""PatchCore-style unsupervised anomaly detector.

Trained only on "good" images (no labels needed). Each image is described
by a grid of locally-aware patch features; every patch is scored by its
distance to the nearest neighbor in a memory bank of normal patch features
collected from train/good. The image-level anomaly score is the maximum
patch score, and the per-patch scores (reshaped to a grid and upsampled to
the input resolution) form the anomaly heatmap that highlights the
suspected defect region.

Reference: Roth et al., "Towards Total Recall in Industrial Anomaly
Detection" (PatchCore), CVPR 2022. This implementation keeps the core idea
(locally-aware patch features + greedy coreset memory bank + nearest-
neighbor scoring) but omits the paper's optional softmax re-weighting term
for simplicity.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models


@dataclass
class AnomalyResult:
    image_score: float
    anomaly_map: torch.Tensor  # (H, W), upsampled to the input image resolution


class PatchFeatureExtractor(nn.Module):
    """Wraps a frozen, pretrained torchvision backbone and captures
    locally-aware, multi-scale patch feature maps."""

    def __init__(self, backbone: str = "resnet18", layers: Tuple[str, str] = ("layer2", "layer3")):
        super().__init__()
        backbone = backbone.lower()
        if backbone == "resnet18":
            net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        elif backbone == "wide_resnet50_2":
            net = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.DEFAULT)
        else:
            raise ValueError(f"Unsupported backbone '{backbone}'")

        net.eval()
        for param in net.parameters():
            param.requires_grad_(False)

        self.net = net
        self.layers = layers
        self._features = {}
        for name in layers:
            getattr(net, name).register_forward_hook(self._make_hook(name))

    def _make_hook(self, name: str):
        def hook(_module, _input, output):
            self._features[name] = output

        return hook

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns a locally-aware patch feature map of shape (B, C, H, W)."""
        self._features = {}
        self.net(x)

        pooled = []
        target_size = None
        for name in self.layers:
            fmap = self._features[name]
            fmap = F.avg_pool2d(fmap, kernel_size=3, stride=1, padding=1)
            if target_size is None:
                target_size = fmap.shape[-2:]
            elif fmap.shape[-2:] != target_size:
                fmap = F.interpolate(fmap, size=target_size, mode="bilinear", align_corners=False)
            pooled.append(fmap)

        return torch.cat(pooled, dim=1)


def _greedy_coreset(
    features: torch.Tensor, n_select: int, projection_dim: int = 128, seed: int = 42
) -> torch.Tensor:
    """Approximate greedy k-center coreset selection. A Johnson-Lindenstrauss
    random projection speeds up the pairwise distance computations used to
    pick maximally-diverse patches, as in the PatchCore paper. The returned
    features are the original (non-projected) vectors."""
    n_total, dim = features.shape
    if n_select >= n_total:
        return features

    generator = torch.Generator().manual_seed(seed)
    projection_dim = min(projection_dim, dim)
    projection = torch.randn(dim, projection_dim, generator=generator)
    projected = features @ projection

    selected_indices: List[int] = []
    min_distances = torch.full((n_total,), float("inf"))

    current_idx = int(torch.randint(0, n_total, (1,), generator=generator).item())
    for _ in range(n_select):
        selected_indices.append(current_idx)
        dist = torch.cdist(projected, projected[current_idx : current_idx + 1]).squeeze(1)
        min_distances = torch.minimum(min_distances, dist)
        min_distances[current_idx] = -1.0  # never re-select the same patch
        current_idx = int(torch.argmax(min_distances).item())

    return features[selected_indices]


class PatchCoreAnomalyDetector:
    """Unsupervised good-vs-defective detector with heatmap localization."""

    def __init__(
        self,
        backbone: str = "resnet18",
        layers: Tuple[str, str] = ("layer2", "layer3"),
        coreset_ratio: float = 0.1,
        max_coreset_size: int = 2000,
        projection_dim: int = 128,
        device: str = "cpu",
        seed: int = 42,
    ):
        self.device = torch.device(device)
        self.extractor = PatchFeatureExtractor(backbone, layers).to(self.device)
        self.coreset_ratio = coreset_ratio
        self.max_coreset_size = max_coreset_size
        self.projection_dim = projection_dim
        self.seed = seed
        self.memory_bank: Optional[torch.Tensor] = None

    @torch.no_grad()
    def _extract_patches(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, 3, H, W) -> patch features (B*H'*W', C)."""
        images = images.to(self.device)
        feature_map = self.extractor(images)  # (B, C, H', W')
        b, c, h, w = feature_map.shape
        patches = feature_map.permute(0, 2, 3, 1).reshape(b * h * w, c)
        return patches.cpu()

    def fit(self, loader) -> None:
        """Build the memory bank of normal patch features from train/good."""
        all_patches = []
        for images, _labels in loader:
            all_patches.append(self._extract_patches(images))
        all_patches = torch.cat(all_patches, dim=0)

        n_select = min(self.max_coreset_size, max(1, int(len(all_patches) * self.coreset_ratio)))
        self.memory_bank = _greedy_coreset(
            all_patches, n_select=n_select, projection_dim=self.projection_dim, seed=self.seed
        )

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> List[AnomalyResult]:
        """images: (B, 3, H, W) -> one AnomalyResult per image."""
        if self.memory_bank is None:
            raise RuntimeError("Call fit() before predict().")

        image_size = images.shape[-2:]
        feature_map = self.extractor(images.to(self.device))
        b, c, h, w = feature_map.shape
        patches = feature_map.permute(0, 2, 3, 1).reshape(b, h * w, c).cpu()

        results = []
        for i in range(b):
            dists = torch.cdist(patches[i], self.memory_bank)  # (h*w, bank_size)
            nn_dists, _ = dists.min(dim=1)  # nearest-neighbor distance per patch
            image_score = nn_dists.max().item()

            anomaly_map = nn_dists.reshape(1, 1, h, w)
            anomaly_map = F.interpolate(
                anomaly_map, size=image_size, mode="bilinear", align_corners=False
            ).squeeze()
            results.append(AnomalyResult(image_score=image_score, anomaly_map=anomaly_map))

        return results

    def save(self, path) -> None:
        torch.save({"memory_bank": self.memory_bank}, path)

    def load(self, path) -> None:
        state = torch.load(path, map_location=self.device)
        self.memory_bank = state["memory_bank"]
