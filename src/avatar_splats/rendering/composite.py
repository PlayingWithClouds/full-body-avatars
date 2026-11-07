"""The avatar as a game draws it: the textured mesh first, its Gaussians over it.

A Gaussian is drawn unless the mesh hides it: when its centre lies behind the mesh's depth at its
pixel by more than a margin, as a depth test against the mesh's depth buffer does in an engine.
The margin lets Gaussians that sit slightly under their own surface still show. Gaussians lie on
the skin as discs, and one whose skin faces away is culled as the triangle under it would be:
seen from the side, the far side of a nose or an ear is less than a margin behind the near one.
"""

from __future__ import annotations

import roma
import torch

from avatar_splats.cameras import Camera
from avatar_splats.gaussians import render
from avatar_splats.rendering.mesh import MeshImage


def behind_mesh(
    means: torch.Tensor, margins: torch.Tensor, depth: torch.Tensor, camera: Camera
) -> torch.Tensor:
    """``(G,)`` bool: which of ``(G, 3)`` world points lie more than ``(G,)`` margins behind the
    ``(H, W)`` mesh depth."""
    screen = camera.screen(means)
    x, y = screen[:, 0].floor().long(), screen[:, 1].floor().long()
    inside = (x >= 0) & (x < camera.width) & (y >= 0) & (y < camera.height)
    mesh_depth = depth.new_full((len(means),), float("inf"))
    mesh_depth[inside] = depth[y[inside], x[inside]]
    return screen[:, 2] > mesh_depth + margins


def facing_away(means: torch.Tensor, rotations: torch.Tensor, camera: Camera) -> torch.Tensor:
    """``(G,)`` bool: which Gaussians at ``(G, 3)`` world points, turned by ``(G, 4)`` wxyz
    rotations, show the camera their back: the side away from their third axis, the normal of
    the skin they lie on."""
    normals = roma.unitquat_to_rotmat(rotations[:, [1, 2, 3, 0]])[:, :, 2]
    return (normals * (camera.position - means)).sum(-1) <= 0


def composite(
    means: torch.Tensor,
    rotations: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    sh: torch.Tensor,
    margins: torch.Tensor,
    mesh: MeshImage,
    camera: Camera,
    sh_degree: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Splat placed Gaussians (see ``gaussians.render``) over a drawn mesh.

    Args:
        margins: ``(G,)`` how far, in metres, each Gaussian may lie behind the mesh and still
            be drawn.

    Returns:
        ``(H, W, 3)`` image and ``(H, W)`` opacity of the avatar.
    """
    hidden = behind_mesh(means.detach(), margins, mesh.depth, camera)
    shown = ~(hidden | facing_away(means.detach(), rotations.detach(), camera))
    image, alpha, _ = render(
        means, rotations, scales, opacities * shown, sh, camera, sh_degree, mesh.image
    )
    return image, alpha + (1 - alpha) * mesh.mask
