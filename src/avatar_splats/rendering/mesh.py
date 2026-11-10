"""Drawing the avatar's textured mesh, the opaque layer under its Gaussians."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from avatar_splats.body_models.mesh import vertex_normals
from avatar_splats.cameras import Camera
from avatar_splats.rendering.rasterize import Fragments, rasterize


@dataclass(frozen=True)
class MeshImage:
    image: torch.Tensor
    """``(H, W, 3)`` colours, the background where the mesh is not."""
    mask: torch.Tensor
    """``(H, W)`` float: 1 where the mesh is."""
    depth: torch.Tensor
    """``(H, W)`` depth of the mesh, infinite where it is not."""


def sample_texture(texture: torch.Tensor, uvs: torch.Tensor) -> torch.Tensor:
    """``(P, C)`` bilinear samples of an ``(H, W, C)`` texture at ``(P, 2)`` texture coordinates
    (u right, v up from the bottom, as in OBJ files)."""
    grid = torch.stack((2 * uvs[:, 0] - 1, 1 - 2 * uvs[:, 1]), dim=-1)
    samples = F.grid_sample(
        texture.permute(2, 0, 1)[None], grid[None, None], align_corners=False, padding_mode="border"
    )
    return samples[0, :, 0].T


def shade(
    fragments: Fragments, uvs: torch.Tensor, textures_of_faces: torch.Tensor,
    textures: Sequence[torch.Tensor],
) -> torch.Tensor:  # fmt: skip
    """``(P, 3)`` texture colour of each fragment.

    Args:
        uvs: ``(F, 3, 2)`` texture coordinates at each triangle's corners.
        textures_of_faces: ``(F,)`` which of ``textures`` each triangle uses.
        textures: ``(H, W, 3)`` textures; they may differ in size.
    """
    uv = fragments.interpolate(uvs)
    chosen = textures_of_faces[fragments.faces]
    colours = uv.new_zeros(len(uv), 3)
    for index, texture in enumerate(textures):
        using = chosen == index
        colours = colours.index_put((using,), sample_texture(texture, uv[using]))
    return colours


def render_mesh(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    uvs: torch.Tensor,
    textures_of_faces: torch.Tensor,
    textures: Sequence[torch.Tensor],
    camera: Camera,
    background: torch.Tensor,
) -> MeshImage:
    """Draw a textured mesh, posed as ``(V, 3)`` world vertices, from one camera."""
    fragments = rasterize(camera.screen(vertices), faces, camera.width, camera.height)
    colours = shade(fragments, uvs, textures_of_faces, textures)
    mask = fragments.mask()
    image = fragments.image(colours)
    image = torch.where(mask.unsqueeze(-1), image, background)
    return MeshImage(image, mask.float(), fragments.depth_image())


def render_shaded(vertices: torch.Tensor, faces: torch.Tensor, camera: Camera) -> torch.Tensor:
    """``(H, W, 3)`` a mesh in plain grey, lit from the camera, on white: to judge its shape.

    Normals are smoothed over the vertices; both sides of a triangle are lit alike.
    """
    fragments = rasterize(camera.screen(vertices), faces, camera.width, camera.height)
    normals = F.normalize(fragments.interpolate(vertex_normals(vertices, faces)[faces]), dim=-1)
    points = fragments.interpolate(vertices[faces])
    towards = F.normalize(camera.position - points, dim=-1)
    light = (normals * towards).sum(-1, keepdim=True).abs()
    grey = 0.2 + 0.75 * light.expand(-1, 3)
    return fragments.image(grey, 1.0)
