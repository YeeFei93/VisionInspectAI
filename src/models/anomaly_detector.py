"""PatchCore-style unsupervised anomaly detector.

Trained only on "good" images (no labels needed). Each image is described
by a grid of locally-aware patch features; every patch is scored by its
distance to the nearest neighbor in a memory bank of normal patch features
collected from train/good. The image-level anomaly score is the maximum
patch score, and the per-patch scores (reshaped to a grid and upsampled to
the input resolution) form the anomaly heatmap that highlights the
suspected defect region.

Reference: Roth et al., "Towards Total Recall in Industrial Anomaly
Detection" (PatchCore), CVPR 2022. This implementation supports nearest-
neighbor or k-nearest-neighbor patch scoring and the paper's optional
softmax image-score reweighting.
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


def scoring_artifact_suffix(
    num_neighbors: int = 1,
    softmax_reweighting: bool = False,
    reweight_num_neighbors: int = 9,
) -> str:
    suffix = f"_knn{num_neighbors}" if num_neighbors != 1 else ""
    if softmax_reweighting:
        suffix += f"_rw{reweight_num_neighbors}"
    return suffix


def _knn_patch_scores(
    distances: torch.Tensor, num_neighbors: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return mean k-NN patch scores and each patch's closest-bank index."""
    if num_neighbors <= 0:
        raise ValueError("num_neighbors must be positive")
    if num_neighbors > distances.shape[1]:
        raise ValueError("num_neighbors cannot exceed the memory-bank size")
    nearest_distances, nearest_indices = torch.topk(
        distances, k=num_neighbors, dim=1, largest=False, sorted=True
    )
    return nearest_distances.mean(dim=1), nearest_indices[:, 0]


def _softmax_reweighted_score(
    query_patch: torch.Tensor,
    nearest_memory_index: int,
    memory_bank: torch.Tensor,
    patch_score: torch.Tensor,
    num_neighbors: int,
) -> torch.Tensor:
    """Apply PatchCore's neighborhood softmax weight to one image score."""
    if num_neighbors <= 1:
        raise ValueError("reweight_num_neighbors must be greater than 1")
    num_neighbors = min(num_neighbors, len(memory_bank))
    nearest_memory_patch = memory_bank[nearest_memory_index : nearest_memory_index + 1]
    memory_distances = torch.cdist(nearest_memory_patch, memory_bank).squeeze(0)
    neighborhood_indices = torch.topk(
        memory_distances, k=num_neighbors, largest=False
    ).indices
    query_distances = torch.cdist(
        query_patch.reshape(1, -1), memory_bank[neighborhood_indices]
    ).squeeze(0)
    nearest_position = torch.nonzero(
        neighborhood_indices == nearest_memory_index, as_tuple=False
    ).item()
    nearest_probability = torch.softmax(query_distances, dim=0)[nearest_position]
    return (1 - nearest_probability) * patch_score


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
    features: torch.Tensor,
    n_select: int,
    projection_dim: int = 128,
    seed: int = 42,
    projection_method: str = "random",
) -> torch.Tensor:
    """Approximate greedy k-center coreset selection. Distances used to pick
    maximally-diverse patches are computed in a lower-dimensional projected
    space for speed; the returned features are always the original
    (non-projected) vectors.

    `projection_method`:
    - "random" (default, as in the PatchCore paper): a Johnson-Lindenstrauss
      random projection. Cheap and data-independent.
    - "pca": a PCA projection onto the top `projection_dim` principal
      components (via `torch.pca_lowrank`), fit on `features` itself.
      Experimental/for-comparison only — unlike the random projection this
      is data-dependent (costs an extra SVD) and, being variance-maximizing
      rather than distance-preserving, is not guaranteed to preserve the
      pairwise distances the greedy k-center selection relies on as well as
      the JL random projection does.
    """
    n_total, dim = features.shape
    if n_select >= n_total:
        return features

    generator = torch.Generator().manual_seed(seed)
    projection_dim = min(projection_dim, dim)
    if projection_method == "pca":
        torch.manual_seed(seed)
        _u, _s, v = torch.pca_lowrank(features, q=projection_dim)
        projected = features @ v
    elif projection_method == "random":
        projection = torch.randn(dim, projection_dim, generator=generator)
        projected = features @ projection
    else:
        raise ValueError(f"Unsupported projection_method '{projection_method}'. Choose 'random' or 'pca'.")

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
        projection_method: str = "random",
        num_neighbors: int = 1,
        softmax_reweighting: bool = False,
        reweight_num_neighbors: int = 9,
    ):
        self.device = torch.device(device)
        self.extractor = PatchFeatureExtractor(backbone, layers).to(self.device)
        self.coreset_ratio = coreset_ratio
        self.max_coreset_size = max_coreset_size
        self.projection_dim = projection_dim
        self.seed = seed
        self.projection_method = projection_method
        if num_neighbors <= 0:
            raise ValueError("num_neighbors must be positive")
        if reweight_num_neighbors <= 1:
            raise ValueError("reweight_num_neighbors must be greater than 1")
        self.num_neighbors = num_neighbors
        self.softmax_reweighting = softmax_reweighting
        self.reweight_num_neighbors = reweight_num_neighbors
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
            all_patches,
            n_select=n_select,
            projection_dim=self.projection_dim,
            seed=self.seed,
            projection_method=self.projection_method,
        )

    @torch.no_grad()
    def predict(
        self, images: torch.Tensor, foreground_masks: Optional[torch.Tensor] = None
    ) -> List[AnomalyResult]:
        """images: (B, 3, H, W) -> one AnomalyResult per image.

        `foreground_masks`, if given, is a (B, H, W) boolean/float tensor at
        the same resolution as `images` (True/1 = object, False/0 =
        background). Background patches are pinned to the minimum
        foreground distance so they read as "normal" in both the image
        score and the heatmap — this keeps the anomaly score and heatmap
        focused on the actual part instead of the plain background.
        """
        if self.memory_bank is None:
            raise RuntimeError("Call fit() before predict().")

        image_size = images.shape[-2:]
        feature_map = self.extractor(images.to(self.device))
        b, c, h, w = feature_map.shape
        patches = feature_map.permute(0, 2, 3, 1).reshape(b, h * w, c).cpu()

        patch_masks = None
        if foreground_masks is not None:
            # Downsample the full-resolution foreground mask to the patch grid.
            patch_masks = (
                F.adaptive_max_pool2d(foreground_masks.float().unsqueeze(1), output_size=(h, w))
                .squeeze(1)
                .reshape(b, h * w)
                > 0
            )

        results = []
        for i in range(b):
            dists = torch.cdist(patches[i], self.memory_bank)  # (h*w, bank_size)
            patch_scores, nearest_indices = _knn_patch_scores(dists, self.num_neighbors)

            if patch_masks is not None and patch_masks[i].any():
                mask_i = patch_masks[i]
                background_fill = patch_scores[mask_i].min()
                patch_scores = torch.where(mask_i, patch_scores, background_fill)
                image_patch_index = int(
                    torch.where(mask_i, patch_scores, torch.tensor(float("-inf"))).argmax().item()
                )
            else:
                image_patch_index = int(patch_scores.argmax().item())

            image_score = patch_scores[image_patch_index]
            if self.softmax_reweighting:
                image_score = _softmax_reweighted_score(
                    patches[i, image_patch_index],
                    int(nearest_indices[image_patch_index].item()),
                    self.memory_bank,
                    image_score,
                    self.reweight_num_neighbors,
                )

            anomaly_map = patch_scores.reshape(1, 1, h, w)
            anomaly_map = F.interpolate(
                anomaly_map, size=image_size, mode="bilinear", align_corners=False
            ).squeeze()
            results.append(AnomalyResult(image_score=image_score.item(), anomaly_map=anomaly_map))

        return results

    def save(self, path) -> None:
        torch.save({"memory_bank": self.memory_bank}, path)

    def load(self, path) -> None:
        state = torch.load(path, map_location=self.device)
        self.memory_bank = state["memory_bank"]
