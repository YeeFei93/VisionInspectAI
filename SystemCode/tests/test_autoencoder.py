import numpy as np
import pytest
import torch
from PIL import Image

from src.evaluation.metrics import compute_per_defect_type_detection
from src.models.autoencoder import (
    ConvAutoencoder,
    gaussian_blur,
    image_score,
    reconstruction_error_map,
    reconstruction_loss,
    ssim_map,
)
from src.preprocessing.transform import get_autoencoder_transforms


def test_autoencoder_reconstructs_input_shape_in_unit_range():
    model = ConvAutoencoder(image_size=64, latent_dim=8, base_channels=4).eval()
    x = torch.rand(2, 3, 64, 64)

    with torch.no_grad():
        out = model(x)

    assert out.shape == x.shape
    assert out.min() >= 0 and out.max() <= 1


def test_autoencoder_rejects_image_size_not_divisible_by_32():
    with pytest.raises(ValueError, match="divisible"):
        ConvAutoencoder(image_size=100)


def test_ssim_of_identical_images_is_one():
    x = torch.rand(1, 3, 32, 32)

    assert torch.allclose(ssim_map(x, x), torch.ones(1, 32, 32), atol=1e-4)


@pytest.mark.parametrize("metric", ["l2", "ssim"])
def test_error_map_is_zero_for_perfect_reconstruction(metric):
    x = torch.rand(2, 3, 32, 32)

    error = reconstruction_error_map(x, x.clone(), metric=metric, blur_sigma=2.0)

    assert error.shape == (2, 32, 32)
    assert error.abs().max() < 1e-4
    assert reconstruction_loss(x, x.clone(), metric).abs() < 1e-4


@pytest.mark.parametrize("metric", ["l2", "ssim"])
def test_error_map_peaks_at_corrupted_region(metric):
    torch.manual_seed(0)
    x = torch.rand(1, 3, 32, 32)
    x_hat = x.clone()
    x_hat[:, :, 20:28, 20:28] = 1.0 - x_hat[:, :, 20:28, 20:28]

    error = reconstruction_error_map(x, x_hat, metric=metric)[0]

    assert error[20:28, 20:28].mean() > 10 * error[:12, :12].mean()


def test_gaussian_blur_preserves_constant_image():
    x = torch.full((1, 1, 16, 16), 0.5)

    assert torch.allclose(gaussian_blur(x, 2.0), x, atol=1e-6)


def test_image_score_reductions():
    anomaly_map = torch.zeros(10, 10)
    anomaly_map[0, 0] = 1.0

    assert image_score(anomaly_map, "mean").item() == pytest.approx(0.01)
    assert image_score(anomaly_map, "max").item() == pytest.approx(1.0)
    assert image_score(anomaly_map, "topk", topk_fraction=0.02).item() == pytest.approx(0.5)
    batched = image_score(torch.stack([anomaly_map, anomaly_map * 2]), "max")
    assert batched.tolist() == pytest.approx([1.0, 2.0])
    with pytest.raises(ValueError):
        image_score(anomaly_map, "median")


def test_autoencoder_transform_keeps_raw_unit_pixels():
    image = Image.new("RGB", (300, 200), color=(255, 0, 128))

    tensor = get_autoencoder_transforms(64)(image)

    assert tensor.shape == (3, 64, 64)
    assert tensor.min() >= 0 and tensor.max() <= 1
    assert tensor[0].mean().item() == pytest.approx(1.0)


def test_per_defect_type_detection_reports_recall_and_auroc():
    defect_types = ["good", "good", "crack", "crack", "cut"]
    labels = [0, 0, 1, 1, 1]
    scores = [0.1, 0.2, 0.9, 0.15, 0.05]
    predictions = [0, 0, 1, 0, 0]

    report = compute_per_defect_type_detection(defect_types, labels, scores, predictions)

    assert report["good"] == {"count": 2, "flagged_rate": 0.0}
    assert report["crack"]["flagged_rate"] == pytest.approx(0.5)
    assert report["crack"]["auroc_vs_good"] == pytest.approx(0.75)
    assert report["cut"]["auroc_vs_good"] == pytest.approx(0.0)
    assert np.isfinite(report["cut"]["flagged_rate"])
