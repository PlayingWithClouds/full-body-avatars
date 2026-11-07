"""Rendering placed Gaussians with gsplat."""

from __future__ import annotations

import gsplat
import torch

from avatar_splats.cameras import Camera


def render(
    means: torch.Tensor,
    rotations: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    sh: torch.Tensor,
    camera: Camera,
    sh_degree: int,
    background: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Render ``G`` Gaussians from one camera.

    Args:
        means: ``(G, 3)`` world positions.
        rotations: ``(G, 4)`` wxyz quaternions.
        scales: ``(G, 3)`` extents along the Gaussians' axes.
        opacities: ``(G,)`` in [0, 1].
        sh: ``(G, K, 3)`` spherical harmonic colour coefficients.
        sh_degree: how many SH bands to use, up to the ones ``sh`` holds.
        background: ``(3,)`` colour behind the Gaussians.

    Returns:
        ``(H, W, 3)`` image, ``(H, W)`` opacity, and gsplat's metadata.
    """
    images, alphas, meta = gsplat.rasterization(
        means=means,
        quats=rotations,
        scales=scales,
        opacities=opacities,
        colors=sh,
        viewmats=camera.view[None],
        Ks=camera.K[None],
        width=camera.width,
        height=camera.height,
        sh_degree=sh_degree,
    )
    image = images[0] + (1 - alphas[0]) * background
    return image.clamp(0, 1), alphas[0, ..., 0], meta
