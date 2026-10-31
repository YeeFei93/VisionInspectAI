"""Embedding-based RBF, GRNN, and SOM classifiers for comparisons."""

from collections import Counter
from typing import Optional

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torchvision import models


class ResNet18EmbeddingExtractor:
    """Extract fixed 512-dimensional ImageNet ResNet18 embeddings."""

    def __init__(self, device: torch.device, pretrained: bool = True):
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        self.model = torch.nn.Sequential(*list(backbone.children())[:-1]).to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def transform(self, loader) -> tuple[np.ndarray, np.ndarray]:
        features, labels = [], []
        for images, batch_labels in loader:
            embeddings = self.model(images.to(self.device)).flatten(1)
            features.append(embeddings.cpu().numpy())
            labels.append(batch_labels.numpy())
        return np.concatenate(features), np.concatenate(labels)


class RBFNetworkClassifier:
    """Multiclass Gaussian RBF network fitted with ridge regression."""

    def __init__(
        self,
        n_centers: int = 64,
        gamma: Optional[float] = None,
        ridge: float = 1e-3,
        random_state: int = 42,
    ):
        self.n_centers = n_centers
        self.gamma = gamma
        self.ridge = ridge
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.pca = None
        self.centers = None
        self.weights = None
        self.classes_ = None

    def _features(self, inputs: np.ndarray) -> np.ndarray:
        scaled = self.scaler.transform(inputs)
        return self.pca.transform(scaled) if self.pca is not None else scaled

    def _rbf(self, inputs: np.ndarray) -> np.ndarray:
        distances = ((inputs[:, None, :] - self.centers[None, :, :]) ** 2).sum(axis=2)
        return np.exp(-self.gamma * distances)

    def fit(self, inputs: np.ndarray, labels: np.ndarray) -> "RBFNetworkClassifier":
        inputs = np.asarray(inputs, dtype=np.float32)
        labels = np.asarray(labels)
        scaled = self.scaler.fit_transform(inputs)
        self.classes_ = np.unique(labels)
        cluster_count = min(self.n_centers, len(scaled))
        kmeans = KMeans(n_clusters=cluster_count, random_state=self.random_state, n_init=10)
        self.centers = kmeans.fit(scaled).cluster_centers_
        if self.gamma is None:
            center_distances = ((self.centers[:, None, :] - self.centers[None, :, :]) ** 2).sum(axis=2)
            nonzero_distances = center_distances[center_distances > 0]
            median_distance = np.median(nonzero_distances) if len(nonzero_distances) else 1.0
            self.gamma = 1.0 / max(median_distance, 1e-6)
        activations = self._rbf(scaled)
        targets = np.eye(len(self.classes_))[np.searchsorted(self.classes_, labels)]
        regularized = activations.T @ activations + self.ridge * np.eye(cluster_count)
        self.weights = np.linalg.solve(regularized, activations.T @ targets)
        return self

    def predict_proba(self, inputs: np.ndarray) -> np.ndarray:
        activations = self._rbf(self._features(np.asarray(inputs, dtype=np.float32)))
        scores = activations @ self.weights
        scores -= scores.max(axis=1, keepdims=True)
        probabilities = np.exp(scores)
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        return self.classes_[self.predict_proba(inputs).argmax(axis=1)]


class GRNNClassifier:
    """Multiclass generalized regression neural network using Gaussian voting."""

    def __init__(self, bandwidth: float = 1.0, pca_components: Optional[int] = 64):
        self.bandwidth = bandwidth
        self.pca_components = pca_components
        self.scaler = StandardScaler()
        self.pca = None
        self.inputs = None
        self.targets = None
        self.classes_ = None

    def _features(self, inputs: np.ndarray) -> np.ndarray:
        scaled = self.scaler.transform(inputs)
        return self.pca.transform(scaled) if self.pca is not None else scaled

    def fit(self, inputs: np.ndarray, labels: np.ndarray) -> "GRNNClassifier":
        inputs = np.asarray(inputs, dtype=np.float32)
        labels = np.asarray(labels)
        scaled = self.scaler.fit_transform(inputs)
        if self.pca_components is not None:
            components = min(self.pca_components, scaled.shape[0], scaled.shape[1])
            self.pca = PCA(n_components=components, random_state=42)
            scaled = self.pca.fit_transform(scaled)
        self.inputs = scaled.astype(np.float32)
        self.classes_ = np.unique(labels)
        self.targets = np.eye(len(self.classes_))[np.searchsorted(self.classes_, labels)]
        return self

    def predict_proba(self, inputs: np.ndarray) -> np.ndarray:
        query = self._features(np.asarray(inputs, dtype=np.float32)).astype(np.float32)
        distances = ((query[:, None, :] - self.inputs[None, :, :]) ** 2).sum(axis=2)
        weights = np.exp(-distances / (2.0 * self.bandwidth**2))
        weights_sum = weights.sum(axis=1, keepdims=True)
        return (weights @ self.targets) / np.maximum(weights_sum, 1e-12)

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        return self.classes_[self.predict_proba(inputs).argmax(axis=1)]


class SOMClassifier:
    """A label-mapped self-organizing map for embedding classification.

    The map is trained without labels. After training, each unit receives the
    majority label of its training samples; validation samples are classified
    by their best-matching unit. Quantization error can also be used as an
    unsupervised anomaly score when the map is trained on normal samples only.
    """

    def __init__(
        self,
        map_rows: int = 4,
        map_cols: int = 4,
        epochs: int = 40,
        learning_rate: float = 0.4,
        sigma: Optional[float] = None,
        pca_components: Optional[int] = 64,
        random_state: int = 42,
    ):
        self.map_rows = map_rows
        self.map_cols = map_cols
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.pca_components = pca_components
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.pca = None
        self.weights = None
        self.node_labels = None
        self.classes_ = None

    @property
    def node_count(self) -> int:
        return self.map_rows * self.map_cols

    def _features(self, inputs: np.ndarray) -> np.ndarray:
        scaled = self.scaler.transform(inputs)
        return self.pca.transform(scaled) if self.pca is not None else scaled

    def _bmu(self, inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        distances = ((inputs[:, None, :] - self.weights[None, :, :]) ** 2).sum(axis=2)
        indices = distances.argmin(axis=1)
        return indices, distances[np.arange(len(inputs)), indices]

    def fit(self, inputs: np.ndarray, labels: Optional[np.ndarray] = None) -> "SOMClassifier":
        inputs = np.asarray(inputs, dtype=np.float32)
        scaled = self.scaler.fit_transform(inputs)
        if self.pca_components is not None:
            components = min(self.pca_components, scaled.shape[0], scaled.shape[1])
            self.pca = PCA(n_components=components, random_state=self.random_state)
            scaled = self.pca.fit_transform(scaled)

        rng = np.random.default_rng(self.random_state)
        initial_indices = rng.choice(len(scaled), size=self.node_count, replace=len(scaled) < self.node_count)
        self.weights = scaled[initial_indices].copy()
        coordinates = np.array(
            [(row, column) for row in range(self.map_rows) for column in range(self.map_cols)],
            dtype=np.float32,
        )
        initial_sigma = self.sigma or max(self.map_rows, self.map_cols) / 2.0

        for epoch in range(self.epochs):
            order = rng.permutation(len(scaled))
            progress = epoch / max(self.epochs - 1, 1)
            rate = self.learning_rate * (1.0 - progress)
            sigma = max(initial_sigma * (1.0 - progress), 0.5)
            for index in order:
                distances = ((self.weights - scaled[index]) ** 2).sum(axis=1)
                bmu = int(distances.argmin())
                map_distances = ((coordinates - coordinates[bmu]) ** 2).sum(axis=1)
                neighborhood = np.exp(-map_distances / (2.0 * sigma**2))
                self.weights += rate * neighborhood[:, None] * (scaled[index] - self.weights)

        if labels is not None:
            labels = np.asarray(labels)
            self.classes_ = np.unique(labels)
            bmus, _ = self._bmu(scaled)
            global_counts = Counter(labels.tolist())
            fallback = max(global_counts, key=global_counts.get)
            self.node_labels = np.full(self.node_count, fallback, dtype=labels.dtype)
            for node in range(self.node_count):
                node_values = labels[bmus == node]
                if len(node_values):
                    self.node_labels[node] = Counter(node_values.tolist()).most_common(1)[0][0]
        return self

    def quantization_error(self, inputs: np.ndarray) -> np.ndarray:
        features = self._features(np.asarray(inputs, dtype=np.float32))
        _, errors = self._bmu(features)
        return np.sqrt(errors)

    def predict_proba(self, inputs: np.ndarray) -> np.ndarray:
        if self.node_labels is None or self.classes_ is None:
            raise RuntimeError("SOMClassifier must be fitted with labels before classification")
        features = self._features(np.asarray(inputs, dtype=np.float32))
        bmus, _ = self._bmu(features)
        probabilities = np.zeros((len(features), len(self.classes_)), dtype=np.float64)
        class_indices = {label: index for index, label in enumerate(self.classes_)}
        for row, node in enumerate(bmus):
            probabilities[row, class_indices[self.node_labels[node]]] = 1.0
        return probabilities

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        return self.classes_[self.predict_proba(inputs).argmax(axis=1)]

    def u_matrix(self) -> np.ndarray:
        """Return mean neighboring weight distance for SOM visualization."""
        grid = self.weights.reshape(self.map_rows, self.map_cols, -1)
        result = np.zeros((self.map_rows, self.map_cols), dtype=np.float32)
        for row in range(self.map_rows):
            for column in range(self.map_cols):
                neighbors = []
                for row_offset, column_offset in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    neighbor_row, neighbor_column = row + row_offset, column + column_offset
                    if 0 <= neighbor_row < self.map_rows and 0 <= neighbor_column < self.map_cols:
                        neighbors.append(np.linalg.norm(grid[row, column] - grid[neighbor_row, neighbor_column]))
                result[row, column] = np.mean(neighbors)
        return result