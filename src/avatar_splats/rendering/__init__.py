"""Drawing the avatar: its textured mesh, and its Gaussians over it."""

from avatar_splats.rendering.composite import behind_mesh, composite, facing_away
from avatar_splats.rendering.mesh import MeshImage, render_mesh, sample_texture, shade
from avatar_splats.rendering.rasterize import Fragments, rasterize

__all__ = [
    "Fragments",
    "MeshImage",
    "behind_mesh",
    "composite",
    "facing_away",
    "rasterize",
    "render_mesh",
    "sample_texture",
    "shade",
]
