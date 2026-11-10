"""A mesh's surface normals and depth as a camera sees them, rasterized by nvdiffrast, to compare
with the surface maps estimated from the frames (``preprocess.surface``)."""

from __future__ import annotations

from dataclasses import dataclass

import nvdiffrast.torch as dr
import torch
import torch.nn.functional as F

from avatar_splats.body_models.mesh import vertex_normals
from avatar_splats.cameras import Camera


@dataclass(frozen=True)
class SurfaceImage:
    covered: torch.Tensor
    """``(B, H, W)`` 1 where the mesh is drawn, else 0."""
    normals: torch.Tensor
    """``(B, H, W, 3)`` unit normals in the camera's axes (x right, y down, z forward)."""
    depth: torch.Tensor
    """``(B, H, W)`` metres along the view direction; 0 where nothing is drawn."""


def clip_space(vertices: torch.Tensor, camera: Camera, near: float, far: float) -> torch.Tensor:
    """``(B, V, 4)`` nvdiffrast clip coordinates of ``(B, V, 3)`` world vertices: image row 0
    on top, as the frames."""
    view = vertices @ camera.view[:3, :3].T + camera.view[:3, 3]
    x, y, z = view.unbind(-1)
    K, width, height = camera.K, camera.width, camera.height
    return torch.stack(
        (
            2 * K[0, 0] / width * x + (2 * K[0, 2] / width - 1) * z,
            2 * K[1, 1] / height * y + (2 * K[1, 2] / height - 1) * z,
            (far + near) / (far - near) * z - 2 * far * near / (far - near),
            z,
        ),
        dim=-1,
    ).contiguous()


def surface_values(
    rast: torch.Tensor, vertices: torch.Tensor, faces: torch.Tensor, camera: Camera
) -> SurfaceImage:
    """Normals and depth of ``(B, V, 3)`` world vertices on an nvdiffrast rasterization of
    their ``(F, 3)`` faces."""
    rotation = camera.view[:3, :3]
    normals = vertex_normals(vertices, faces) @ rotation.T
    depth = vertices @ rotation[2] + camera.view[2, 3]
    values, _ = dr.interpolate(
        torch.cat((normals, depth[..., None]), dim=-1).contiguous(), rast, faces.int().contiguous()
    )
    covered = (rast[..., 3] > 0).float()
    return SurfaceImage(
        covered=covered,
        normals=F.normalize(values[..., :3], dim=-1),
        depth=values[..., 3] * covered,
    )


def render_surface(
    context: dr.RasterizeCudaContext,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    camera: Camera,
    near: float,
    far: float,
) -> SurfaceImage:
    """Normals and depth of ``(B, V, 3)`` world vertices of a mesh with ``(F, 3)`` faces."""
    clip = clip_space(vertices, camera, near, far)
    rast, _ = dr.rasterize(context, clip, faces.int().contiguous(), (camera.height, camera.width))
    return surface_values(rast, vertices, faces, camera)
