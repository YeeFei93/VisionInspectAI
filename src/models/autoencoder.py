"""Convolutional autoencoder for reconstruction-based anomaly detection.

A second unsupervised detector to compare against PatchCore: the network
is trained only on `train/good` images to reconstruct normal products, so
at test time defective regions are reconstructed poorly ("what a normal
part should look like") and the per-pixel reconstruction error doubles as
an anomaly heatmap. The design follows the MVTec-AD reference autoencoder
(Bergmann et al.): a strided-conv encoder squeezed into a small 1x1 latent
vector, so the model cannot simply learn an identity map and also
reconstruct the defect.

The helpers below (`ssim_map`, `reconstruction_error_map`, `image_score`)
are pure tensor functions so they can be unit-tested without a checkpoint.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

ERROR_METRICS = ("l2", "ssim")
SCORE_REDUCTIONS = ("mean", "max", "topk")


def _down_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2, inplace=True),
    )


def _up_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2, inplace=True),
    )


class ConvAutoencoder(nn.Module):
    """224x224 RGB -> (latent_dim x 1 x 1) -> 224x224 RGB in [0, 1].

    Five stride-2 down blocks take 224 -> 7, a 7x7 conv collapses that to a
    1x1 latent vector, and the decoder mirrors the path back up."""

    def __init__(
        self,
        image_size: int = 224,
        latent_dim: int = 128,
        base_channels: int = 32,
        in_channels: int = 3,
    ):
        super().__init__()
        num_down = 5
        if image_size % (2 ** num_down) != 0:
            raise ValueError(f"image_size must be divisible by {2 ** num_down}")
        bottleneck_size = image_size // (2 ** num_down)
        channels = [base_channels * (2 ** min(i, 3)) for i in range(num_down)]

        encoder_layers = []
        previous = in_channels
        for out_channels in channels:
            encoder_layers.append(_down_block(previous, out_channels))
            previous = out_channels
        encoder_layers.append(nn.Conv2d(previous, latent_dim, kernel_size=bottleneck_size))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = [
            nn.ConvTranspose2d(latent_dim, previous, kernel_size=bottleneck_size),
            nn.BatchNorm2d(previous),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        for out_channels in reversed(channels[:-1]):
            decoder_layers.append(_up_block(previous, out_channels))
            previous = out_channels
        decoder_layers.append(
            nn.ConvTranspose2d(previous, in_channels, kernel_size=4, stride=2, padding=1)
        )
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)

        self.image_size = image_size
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def _gaussian_kernel(sigma: float, channels: int, device, dtype) -> torch.Tensor:
    radius = max(1, int(math.ceil(3 * sigma)))
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d.expand(channels, 1, -1, -1).contiguous()


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Depthwise Gaussian blur of an (N, C, H, W) tensor (reflect padding)."""
    if sigma <= 0:
        return x
    kernel = _gaussian_kernel(sigma, x.shape[1], x.device, x.dtype)
    pad = kernel.shape[-1] // 2
    padded = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    return F.conv2d(padded, kernel, groups=x.shape[1])


def ssim_map(
    x: torch.Tensor,
    y: torch.Tensor,
    sigma: float = 1.5,
    data_range: float = 1.0,
) -> torch.Tensor:
    """Per-pixel SSIM between two (N, C, H, W) images, averaged over
    channels -> (N, H, W). 1 means locally identical structure."""
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_x = gaussian_blur(x, sigma)
    mu_y = gaussian_blur(y, sigma)
    sigma_x = gaussian_blur(x * x, sigma) - mu_x ** 2
    sigma_y = gaussian_blur(y * y, sigma) - mu_y ** 2
    sigma_xy = gaussian_blur(x * y, sigma) - mu_x * mu_y
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    return (numerator / denominator).mean(dim=1)


def reconstruction_loss(x: torch.Tensor, x_hat: torch.Tensor, metric: str = "l2") -> torch.Tensor:
    if metric == "l2":
        return F.mse_loss(x_hat, x)
    if metric == "ssim":
        return 1.0 - ssim_map(x, x_hat).mean()
    raise ValueError(f"metric must be one of {ERROR_METRICS}")


def reconstruction_error_map(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    metric: str = "l2",
    blur_sigma: float = 0.0,
) -> torch.Tensor:
    """Per-pixel anomaly map (N, H, W): squared error averaged over RGB for
    `l2`, or 1 - SSIM for `ssim`. Optional Gaussian smoothing suppresses
    single-pixel reconstruction noise, like PatchCore's map smoothing."""
    if metric == "l2":
        error = ((x - x_hat) ** 2).mean(dim=1)
    elif metric == "ssim":
        error = 1.0 - ssim_map(x, x_hat)
    else:
        raise ValueError(f"metric must be one of {ERROR_METRICS}")
    if blur_sigma > 0:
        error = gaussian_blur(error.unsqueeze(1), blur_sigma).squeeze(1)
    return error


def image_score(
    anomaly_map: torch.Tensor,
    reduction: str = "mean",
    topk_fraction: float = 0.01,
) -> torch.Tensor:
    """Collapse an (H, W) or (N, H, W) anomaly map into one score per image.
    `mean` dilutes small defects over the whole frame; `max` reacts to the
    single worst pixel; `topk` averages the worst `topk_fraction` of pixels
    as a middle ground."""
    flat = anomaly_map.reshape(anomaly_map.shape[0], -1) if anomaly_map.dim() == 3 else anomaly_map.reshape(1, -1)
    if reduction == "mean":
        scores = flat.mean(dim=1)
    elif reduction == "max":
        scores = flat.max(dim=1).values
    elif reduction == "topk":
        if not 0.0 < topk_fraction <= 1.0:
            raise ValueError("topk_fraction must be in (0, 1]")
        k = max(1, int(round(flat.shape[1] * topk_fraction)))
        scores = flat.topk(k, dim=1).values.mean(dim=1)
    else:
        raise ValueError(f"reduction must be one of {SCORE_REDUCTIONS}")
    return scores if anomaly_map.dim() == 3 else scores.squeeze(0)
