"""How much the avatar's colours change as it is seen from different directions.

A surface point should look about the same from every side. Overlapping Gaussians of different
colours break that: which one shows depends on the order they are blended in, and the avatar
shimmers as it turns. The training frames cannot tell, since they are the very views the
Gaussians were fitted to, all from one eye-level orbit. So the avatar is drawn from all around,
above and below eye level too, and each visible point's colour is compared across the views
that see it squarely: the spread of its colour across them.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from avatar_splats.body_models.hybrid import BODY
from avatar_splats.cameras import Camera
from avatar_splats.rendering import rasterize, sample_texture
from avatar_splats.training.views import View

YAWS = tuple(range(0, 360, 30))
PITCHES = (-30, 0, 30)
"""Viewing directions in degrees around and above the avatar."""
MIN_COSINE = 0.5
"""A view counts for a point when it sees the surface within 60 degrees of face on."""
DEPTH_TOLERANCE = 0.01


@torch.no_grad()
def view_spread(
    scene, vertices: torch.Tensor, camera: Camera, occlusion_margin: float,
    parts: Sequence[int] = (BODY,), points: int = 3000, seed: int = 0,
) -> torch.Tensor:  # fmt: skip
    """``(P,)`` for points on the surface of some of the avatar's parts, seen squarely from two
    views or more, the standard deviation of their colour across those views (mean over
    channels, 0 to 1).

    Args:
        scene: a ``train.Scene``.
        vertices: ``(V, 3)`` the posed avatar.
        camera: lens and image size to draw with; the views frame the chosen parts.
        parts: the parts to sample, from ``body_models.hybrid``.
    """
    device = vertices.device
    faces = scene.faces
    face_parts = torch.from_numpy(scene.avatar.face_parts).to(device)
    chosen_faces = torch.nonzero(torch.isin(face_parts, face_parts.new_tensor(parts))).squeeze(-1)
    corners = vertices[faces[chosen_faces]]
    areas = torch.linalg.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    generator = torch.Generator(device=device).manual_seed(seed)
    chosen = torch.multinomial(areas.norm(dim=-1), points, replacement=True, generator=generator)
    u, v = torch.rand(2, points, device=device, generator=generator)
    flip = u + v > 1
    u, v = torch.where(flip, 1 - u, u), torch.where(flip, 1 - v, v)
    weights = torch.stack((1 - u - v, u, v), dim=-1)
    samples = (weights.unsqueeze(-1) * corners[chosen]).sum(dim=1)
    normals = torch.nn.functional.normalize(areas[chosen], dim=-1)

    centre = (corners.amin(dim=(0, 1)) + corners.amax(dim=(0, 1))) / 2
    height = float((corners.amax(dim=(0, 1)) - corners.amin(dim=(0, 1))).max())
    distance = 1.2 * height / 2 * float(camera.K[1, 1]) / (camera.height / 2)

    sums = torch.zeros(points, 3, device=device)
    squares = torch.zeros(points, 3, device=device)
    views = torch.zeros(points, device=device)
    for pitch in PITCHES:
        for yaw in YAWS:
            p, y = math.radians(pitch), math.radians(yaw)
            direction = torch.tensor(
                [math.sin(y) * math.cos(p), -math.cos(y) * math.cos(p), math.sin(p)],
                device=device,
            )
            view_camera = camera.looking_at(centre + distance * direction, centre)
            view = View("body", 0, 0, None, None, None, view_camera)
            colour, _ = scene.render(vertices, view, scene_degree(scene), occlusion_margin)
            screen = view_camera.screen(samples)
            depth = rasterize(
                view_camera.screen(vertices), faces, view_camera.width, view_camera.height
            ).depth_image()
            x, y_ = screen[:, 0].floor().long(), screen[:, 1].floor().long()
            inside = (x >= 0) & (x < view_camera.width) & (y_ >= 0) & (y_ < view_camera.height)
            x, y_ = x.clamp(0, view_camera.width - 1), y_.clamp(0, view_camera.height - 1)
            facing = (normals * direction).sum(-1) >= MIN_COSINE
            seen = inside & facing & (screen[:, 2] <= depth[y_, x] + DEPTH_TOLERANCE)
            uv = torch.stack(
                (screen[:, 0] / view_camera.width, 1 - screen[:, 1] / view_camera.height), dim=-1
            )
            seen_colour = sample_texture(colour, uv) * seen[:, None]
            sums += seen_colour
            squares += seen_colour.square()
            views += seen
    counted = views >= 2
    mean = sums[counted] / views[counted, None]
    variance = (squares[counted] / views[counted, None] - mean.square()).clamp_min(0)
    return variance.sqrt().mean(dim=-1)


def scene_degree(scene) -> int:
    """The highest spherical-harmonic degree the scene's Gaussians have."""
    return math.isqrt(scene.gaussians.sh_rest.shape[1] + 1) - 1
