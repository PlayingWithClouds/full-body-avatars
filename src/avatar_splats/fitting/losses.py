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


def normal_loss(
    rendered: torch.Tensor, targets: torch.Tensor, region: torch.Tensor
) -> torch.Tensor:
    """Mean of one minus the cosine between ``(B, H, W, 3)`` rendered and estimated unit
    normals over a ``(B, H, W)`` region: 0 where they agree, 2 where they are opposed."""
    cosine = (rendered * targets).sum(-1)
    return ((1 - cosine) * region).sum() / region.sum().clamp_min(1)


def depth_loss(
    rendered: torch.Tensor, targets: torch.Tensor, region: torch.Tensor, cap: float = 0.3
) -> torch.Tensor:
    """Mean absolute difference, in metres, between ``(B, H, W)`` rendered depth and estimated
    depth over a ``(B, H, W)`` region, the estimate scaled per image so that the medians of the
    two agree there.

    The estimate's scale is its own guess, and a body of the wrong size or at the wrong
    distance would shift it as a whole; what stays is the relief: which part is in front and by
    how much. Differences are capped at ``cap`` metres, so wrong estimates pull no harder than
    that. NaN targets count as outside the region.
    """
    region = region * torch.isfinite(targets)
    targets = torch.where(region > 0, targets, 0)
    with torch.no_grad():
        inside = region > 0
        nan = torch.full_like(rendered, torch.nan)
        drawn = torch.where(inside, rendered, nan).flatten(1).nanmedian(dim=1).values
        estimated = torch.where(inside, targets, nan).flatten(1).nanmedian(dim=1).values
        scale = torch.nan_to_num(drawn / estimated, nan=0.0)
    error = (rendered - scale[:, None, None] * targets).abs().clamp_max(cap)
    return (error * region).sum() / region.sum().clamp_min(1)


def keypoint_depth_loss(
    depth: torch.Tensor,
    targets: torch.Tensor,
    insets: torch.Tensor,
    importance: torch.Tensor | None = None,
    cap: float = 0.1,
) -> torch.Tensor:
    """Mean distance, in metres, of ``(..., K)`` depths of fitted keypoints from estimated
    depths of the skin at their pixels (NaN: unknown), the keypoints lying ``(K,)`` insets
    behind the skin.

    The skin at a keypoint's pixel may be its own or that of whatever hides it, so the term
    is lopsided: a keypoint is pulled onto the skin at most ``cap`` metres' worth (a hidden one
    rightly stays behind), but always pushed back behind it when it comes nearer the camera.
    Not harder than that: the estimates are noisy, and a steeper push back (4x) drove limbs and
    the head behind the body to escape it (IoU 0.83 to 0.66).
    As in ``depth_loss``, each row's estimates are scaled so that their median agrees with the
    body's: what counts is which keypoint is in front and by how much. Rows with fewer than two
    known keypoints count for nothing. ``importance``: ``(K,)`` weights, 1 by default.
    """
    known = torch.isfinite(targets)
    weights = (known & (known.sum(-1, keepdim=True) >= 2)).float()
    if importance is not None:
        weights = weights * importance
    targets = torch.where(known, targets, 0)
    skin = depth - insets
    with torch.no_grad():
        ratio = torch.where(known, skin / targets.clamp_min(1e-6), torch.nan)
        scale = torch.nan_to_num(ratio.nanmedian(dim=-1).values, nan=0.0)
    behind = skin - scale[..., None] * targets  # negative: nearer than the skin
    error = behind.abs().clamp_max(cap) + torch.relu(-behind - cap)
    return (error * weights).sum() / weights.sum().clamp_min(1e-6)


def second_difference(values: torch.Tensor) -> torch.Tensor:
    """``values[i + 1] - 2 values[i] + values[i - 1]`` along axis 0: zero for steady motion."""
    return values[2:] - 2 * values[1:-1] + values[:-2]
