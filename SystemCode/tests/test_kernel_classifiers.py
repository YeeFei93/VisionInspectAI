"""Tests for the model-free parts of the embedding classifiers."""

import numpy as np

from src.models.kernel_classifiers import GRNNClassifier, RBFNetworkClassifier, SOMClassifier


def test_rbf_classifier_fits_and_returns_probabilities():
    inputs = np.array([[0.0, 0.0], [0.0, 1.0], [5.0, 5.0], [5.0, 6.0]])
    labels = np.array([0, 0, 1, 1])

    classifier = RBFNetworkClassifier(n_centers=2).fit(inputs, labels)

    assert classifier.predict(inputs).tolist() == labels.tolist()
    assert np.allclose(classifier.predict_proba(inputs).sum(axis=1), 1.0)


def test_grnn_classifier_fits_and_returns_probabilities():
    inputs = np.array([[0.0, 0.0], [0.0, 1.0], [5.0, 5.0], [5.0, 6.0]])
    labels = np.array([0, 0, 1, 1])

    classifier = GRNNClassifier(bandwidth=1.0, pca_components=None).fit(inputs, labels)

    assert classifier.predict(inputs).tolist() == labels.tolist()
    assert np.allclose(classifier.predict_proba(inputs).sum(axis=1), 1.0)


def test_som_classifier_maps_clusters_to_labels_and_scores_distance():
    inputs = np.array([[0.0, 0.0], [0.0, 1.0], [5.0, 5.0], [5.0, 6.0]])
    labels = np.array([0, 0, 1, 1])

    classifier = SOMClassifier(
        map_rows=2, map_cols=2, epochs=30, pca_components=None
    ).fit(inputs, labels)

    assert classifier.predict(inputs).tolist() == labels.tolist()
    assert classifier.u_matrix().shape == (2, 2)
    assert np.all(classifier.quantization_error(inputs) >= 0)