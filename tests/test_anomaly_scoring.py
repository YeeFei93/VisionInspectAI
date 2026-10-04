import pytest
import torch
from torch import nn

import src.models.anomaly_detector as anomaly_detector_module
from src.models.anomaly_detector import (
    PatchCoreAnomalyDetector,
    _knn_patch_scores,
    _softmax_reweighted_score,
    scoring_artifact_suffix,
)


class FixedExtractor(nn.Module):
    def forward(self, images):
        return torch.tensor([[[[0.2, 2.0]]]], dtype=torch.float32)


def make_detector(num_neighbors=1, softmax_reweighting=False):
    detector = object.__new__(PatchCoreAnomalyDetector)
    detector.extractor = FixedExtractor()
    detector.memory_bank = torch.tensor([[0.0], [1.0], [4.0]])
    detector.num_neighbors = num_neighbors
    detector.softmax_reweighting = softmax_reweighting
    detector.reweight_num_neighbors = 3
    detector.device = torch.device("cpu")
    return detector


def test_knn_patch_scores_average_k_smallest_distances():
    distances = torch.tensor([[4.0, 1.0, 3.0, 2.0], [8.0, 2.0, 5.0, 9.0]])

    scores, nearest_indices = _knn_patch_scores(distances, num_neighbors=3)

    assert torch.allclose(scores, torch.tensor([2.0, 5.0]))
    assert nearest_indices.tolist() == [1, 1]


def test_knn_one_matches_nearest_neighbor_scoring():
    distances = torch.tensor([[2.0, 0.5, 1.0]])

    scores, nearest_indices = _knn_patch_scores(distances, num_neighbors=1)

    assert scores.item() == pytest.approx(0.5)
    assert nearest_indices.item() == 1


def test_softmax_reweighting_uses_nearest_memory_patch_probability():
    memory_bank = torch.tensor([[0.0], [1.0], [3.0]])
    query_patch = torch.tensor([1.2])
    patch_score = torch.tensor(2.0)

    score = _softmax_reweighted_score(
        query_patch,
        nearest_memory_index=1,
        memory_bank=memory_bank,
        patch_score=patch_score,
        num_neighbors=3,
    )

    query_distances = torch.tensor([0.2, 1.2, 1.8])
    expected = (1 - torch.softmax(query_distances, dim=0)[0]) * patch_score
    assert score.item() == pytest.approx(expected.item())


@pytest.mark.parametrize("num_neighbors", [0, -1])
def test_knn_patch_scores_reject_non_positive_k(num_neighbors):
    with pytest.raises(ValueError, match="positive"):
        _knn_patch_scores(torch.ones((1, 3)), num_neighbors)


def test_knn_patch_scores_reject_k_larger_than_memory_bank():
    with pytest.raises(ValueError, match="memory-bank size"):
        _knn_patch_scores(torch.ones((1, 3)), num_neighbors=4)


def test_detector_knn_scoring_changes_patch_map():
    image = torch.zeros((1, 3, 1, 2))

    nearest_map = make_detector(num_neighbors=1).predict(image)[0].anomaly_map
    knn_map = make_detector(num_neighbors=2).predict(image)[0].anomaly_map

    assert not torch.allclose(nearest_map, knn_map)


def test_detector_softmax_reweighting_changes_only_image_score():
    image = torch.zeros((1, 3, 1, 2))

    plain = make_detector().predict(image)[0]
    reweighted = make_detector(softmax_reweighting=True).predict(image)[0]

    assert torch.allclose(plain.anomaly_map, reweighted.anomaly_map)
    assert reweighted.image_score != pytest.approx(plain.image_score)


def test_detector_keeps_full_resolution_background_at_fill_score():
    image = torch.zeros((1, 3, 4, 8))
    foreground_mask = torch.zeros((1, 4, 8), dtype=torch.bool)
    foreground_mask[:, :, :4] = True

    result = make_detector().predict(image, foreground_masks=foreground_mask)[0]

    assert result.image_score == pytest.approx(0.2)
    assert torch.all(result.anomaly_map[:, 4:] == result.anomaly_map[:, 4:].min())
    assert result.anomaly_map[:, :4].max() > result.anomaly_map[:, 4:].max()


def test_fit_caps_coreset_candidates_deterministically(monkeypatch):
    captured_candidates = []

    def capture_coreset(features, n_select, **_kwargs):
        captured_candidates.append(features.clone())
        return features[:n_select]

    monkeypatch.setattr(anomaly_detector_module, "_greedy_coreset", capture_coreset)
    loader = [(torch.arange(20).reshape(10, 2).float(), None) for _ in range(3)]

    for _ in range(2):
        detector = object.__new__(PatchCoreAnomalyDetector)
        detector.seed = 42
        detector.max_coreset_candidates = 9
        detector.max_coreset_size = 3
        detector.coreset_ratio = 1.0
        detector.projection_dim = 2
        detector.projection_method = "random"
        detector._extract_patches = lambda images: images
        detector.fit(loader)

    assert len(captured_candidates[0]) == 9
    assert torch.equal(captured_candidates[0], captured_candidates[1])


@pytest.mark.parametrize(
    ("num_neighbors", "reweighting", "support", "expected"),
    [(1, False, 9, ""), (3, False, 9, "_knn3"), (1, True, 9, "_rw9"), (3, True, 5, "_knn3_rw5")],
)
def test_scoring_artifact_suffix(num_neighbors, reweighting, support, expected):
    assert scoring_artifact_suffix(num_neighbors, reweighting, support) == expected