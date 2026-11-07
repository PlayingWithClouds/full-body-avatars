"""Loss terms shared by the body and head fits."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def keypoint_loss(
    projected: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor, image_height: int
) -> torch.Tensor:
    """Weighted robust distance between ``(..., K, 2)`` projected and detected points.

    Distances are in image heights, with a smooth L1 below 0.5 % of the height, so the loss
    reads as the typical error in image heights and outliers pull no harder than near misses.
    """
    distance = (projected - targets).norm(dim=-1) / image_height
    robust = F.smooth_l1_loss(distance, torch.zeros_like(distance), beta=0.005, reduction="none")
    return (weights * robust).sum() / weights.sum().clamp_min(1e-6)


def silhouette_loss(
    rendered: torch.Tensor, masks: torch.Tensor, region: torch.Tensor | None = None
) -> torch.Tensor:
    """Mean squared difference of ``(B, H, W)`` silhouettes and masks, optionally in a region."""
    error = (rendered - masks).square()
    if region is None:
        return error.mean()
    return (error * region).sum() / region.sum().clamp_min(1)


def second_difference(values: torch.Tensor) -> torch.Tensor:
    """``values[i + 1] - 2 values[i] + values[i - 1]`` along axis 0: zero for steady motion."""
    return values[2:] - 2 * values[1:-1] + values[:-2]
