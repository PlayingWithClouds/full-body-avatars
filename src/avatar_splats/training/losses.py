"""Image losses for training the Gaussians."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def ssim(a: torch.Tensor, b: torch.Tensor, window: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """``(H, W)`` structural similarity map of two ``(H, W, 3)`` images in [0, 1]."""
    coords = torch.arange(window, device=a.device, dtype=a.dtype) - window // 2
    kernel = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel = (kernel / kernel.sum()).outer(kernel / kernel.sum()).expand(3, 1, -1, -1)

    def blur(x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, kernel, padding=window // 2, groups=3)

    a, b = a.permute(2, 0, 1)[None], b.permute(2, 0, 1)[None]
    mu_a, mu_b = blur(a), blur(b)
    var_a = blur(a * a) - mu_a**2
    var_b = blur(b * b) - mu_b**2
    covariance = blur(a * b) - mu_a * mu_b
    c1, c2 = 0.01**2, 0.03**2
    value = ((2 * mu_a * mu_b + c1) * (2 * covariance + c2)) / (
        (mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)
    )
    return value[0].mean(dim=0)


def psnr(a: torch.Tensor, b: torch.Tensor, region: torch.Tensor) -> float:
    """Peak signal-to-noise ratio in dB of ``(H, W, 3)`` images within a ``(H, W)`` region."""
    error = ((a - b).square().mean(-1) * region).sum() / region.sum().clamp_min(1)
    return float(-10 * torch.log10(error.clamp_min(1e-10)))
