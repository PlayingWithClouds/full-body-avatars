"""Differentiable silhouettes of meshes, for fitting them to segmentation masks.

Points spread evenly over the surface are splatted as small round Gaussians; the accumulated
opacity is the silhouette. It is smooth at the edges, so its gradients move the mesh towards the
mask, and it reuses the renderer the avatar is trained with.

The blobs reach past the surface they are drawn from, so a fit that matches their outline to the
mask leaves the mesh too thin: by 15 % of the area on the body, most on thin limbs. The points
are therefore moved inwards along the surface's normal (``inset``), until the blobs' outline is
the mesh's.
"""

from __future__ import annotations

from dataclasses import dataclass

import gsplat
import numpy as np
import torch

from avatar_splats.cameras import Camera

SPLAT_SIZE = 0.8
"""Radius of the splatted blobs, in sample spacings."""


@dataclass(frozen=True)
class SurfaceSamples:
    """Fixed points on a mesh's triangles, which follow the mesh as it deforms."""

    faces: torch.Tensor
    """``(S,)`` the triangle each point is on."""
    barycentric: torch.Tensor
    """``(S, 3)`` the point's barycentric coordinates in its triangle."""
    spacing: float
    """Typical distance between neighbouring points on the mesh the samples were drawn from."""

    @classmethod
    def draw(
        cls, vertices: torch.Tensor, faces: torch.Tensor, count: int, seed: int = 0
    ) -> SurfaceSamples:
        """Draw ``count`` points spread uniformly by area over ``(V, 3)`` vertices' triangles."""
        a, b, c = (vertices[faces[:, i]] for i in range(3))
        areas = torch.linalg.cross(b - a, c - a).norm(dim=-1) / 2
        generator = torch.Generator(device=vertices.device).manual_seed(seed)
        chosen = torch.multinomial(areas, count, replacement=True, generator=generator)
        u, v = torch.rand(2, count, device=vertices.device, generator=generator)
        flip = u + v > 1
        u, v = torch.where(flip, 1 - u, u), torch.where(flip, 1 - v, v)
        barycentric = torch.stack((1 - u - v, u, v), dim=-1)
        return cls(chosen, barycentric, float((areas.sum() / count).sqrt()))

    @property
    def radius(self) -> float:
        """Radius of the blobs splatted at the points: large enough to leave no holes."""
        return SPLAT_SIZE * self.spacing

    def points(
        self, vertices: torch.Tensor, faces: torch.Tensor, inset: float = 0.0
    ) -> torch.Tensor:
        """``(..., S, 3)`` the points on ``(..., V, 3)`` deformed vertices, moved ``inset``
        inwards along their triangles' normals (for outward-facing triangles)."""
        corners = vertices[..., faces[self.faces], :]
        points = (self.barycentric.unsqueeze(-1) * corners).sum(dim=-2)
        if inset:
            a, b, c = corners.unbind(dim=-2)
            normals = torch.nn.functional.normalize(torch.linalg.cross(b - a, c - a), dim=-1)
            points = points - inset * normals
        return points


def render_silhouette(points: torch.Tensor, radius: float, camera: Camera) -> torch.Tensor:
    """Render ``(B, S, 3)`` point sets as ``(B, H, W)`` silhouettes in [0, 1]."""
    batch, count = points.shape[:2]
    ones = points.new_ones(batch, count)
    _, alphas, _ = gsplat.rasterization(
        means=points,
        quats=torch.tensor([1.0, 0, 0, 0], device=points.device).expand(batch, count, 4),
        scales=points.new_full((batch, count, 3), radius),
        opacities=ones,
        colors=ones.unsqueeze(-1),
        viewmats=camera.view.expand(batch, 1, 4, 4),
        Ks=camera.K.expand(batch, 1, 3, 3),
        width=camera.width,
        height=camera.height,
    )
    return alphas[:, 0, ..., 0]


def overlay(mask: np.ndarray, rendered: np.ndarray) -> np.ndarray:
    """``(H, W, 3)`` uint8 comparison of two ``(H, W)`` silhouettes.

    White where both cover the pixel, red where only the mask does, cyan where only the render.
    """
    mask, rendered = mask > 0.5, rendered > 0.5
    image = np.zeros((*mask.shape, 3), np.uint8)
    image[mask & rendered] = 255
    image[mask & ~rendered] = (230, 60, 60)
    image[~mask & rendered] = (60, 210, 240)
    return image


def silhouette_iou(rendered: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """``(B,)`` IoU of thresholded ``(B, H, W)`` silhouettes with masks."""
    inside = rendered > 0.5
    masks = masks > 0.5
    union = (inside | masks).sum(dim=(1, 2)).clamp_min(1)
    return (inside & masks).sum(dim=(1, 2)) / union
