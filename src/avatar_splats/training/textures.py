"""The avatar's textures, baked from the frames: before training, and again as it goes, from the
poses training has corrected.

Each texel is a weighted average of many frames, so a texture baked this way cannot drift into
noise the way one learned texel by texel does: a texel is seen in few frames, and an optimizer
moves the rarely seen ones by as much as the often seen ones.

Baking projects every texel's point on the surface into every frame. Where a frame sees it, not
hidden by other parts of the mesh, inside the performer's mask and facing the camera, the frame
votes for its colour, the more the more squarely and the more closely it sees it. A texel's
coverage is how squarely the best frame saw it: 1 face on, 0 never. Places no frame sees (the
armpits, the soles) are filled from their surroundings, so they show plausible skin.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from avatar_splats.body_models.hybrid import BODY_TEXTURE, HEAD_TEXTURE, Avatar
from avatar_splats.body_models.ict import UV_TILES
from avatar_splats.body_models.mesh import triangle_frames
from avatar_splats.records import ArrayRecord
from avatar_splats.rendering import Fragments, rasterize, sample_texture
from avatar_splats.training.colour import ColourCorrection
from avatar_splats.training.posing import Posing
from avatar_splats.training.views import View

DEPTH_TOLERANCE = 0.01
"""How far, in metres, a texel's point may lie behind the mesh's depth and still count as seen."""
SHARPNESS = 4
"""Power of the cosine of the viewing angle in a frame's vote: oblique views smear the texture."""
MASK_EROSION = 5
"""Pixels this close to the mask's edge, where colours mix with the background, have no vote."""


@dataclass(frozen=True)
class TextureRecord(ArrayRecord):
    body: np.ndarray
    """``(H, W, 3)`` float colours in [0, 1] in Anny's UV layout."""
    head: np.ndarray
    """``(H, 5 H, 3)`` float colours in [0, 1] in the head's atlas."""
    body_coverage: np.ndarray
    """``(H, W)`` float in [0, 1]: how squarely the best frame saw each texel."""
    head_coverage: np.ndarray


def texel_fragments(uvs: torch.Tensor, faces: torch.Tensor, width: int, height: int) -> Fragments:
    """The texels of a ``width`` x ``height`` texture covered by the given triangles.

    Args:
        uvs: ``(F, 3, 2)`` texture coordinates of all the mesh's triangles.
        faces: ``(T,)`` indices of the triangles using this texture.
    """
    corners = uvs[faces].reshape(-1, 2)
    points = torch.stack(
        (corners[:, 0] * width, (1 - corners[:, 1]) * height, torch.ones_like(corners[:, 0])), 1
    )
    layout = torch.arange(len(points), device=uvs.device).view(-1, 3)
    fragments = rasterize(points, layout, width, height)
    return dataclasses.replace(fragments, faces=faces[fragments.faces])


@torch.no_grad()
def bake_textures(
    avatar: Avatar,
    views: list[View],
    posing: Posing,
    body_size: int,
    head_size: int,
    correction: ColourCorrection | None = None,
) -> TextureRecord:
    """Bake the avatar's textures from the views: the body's from the body video, the head's
    from all three. The head texture's tiles are ``head_size`` texels square.

    Args:
        correction: turns each video's colours into the reference video's before they are
            averaged; without it, the videos' colours are taken as they are.
    """
    device = posing.rig.faces.device
    faces = posing.rig.faces
    uvs = torch.from_numpy(avatar.uvs).to(device)
    textures_of_faces = torch.from_numpy(avatar.face_textures).to(device)
    head_faces = torch.nonzero(textures_of_faces == HEAD_TEXTURE).squeeze(-1)

    baked = {}
    for texture, (width, height) in (
        (BODY_TEXTURE, (body_size, body_size)),
        (HEAD_TEXTURE, (UV_TILES * head_size, head_size)),
    ):
        texels = texel_fragments(
            uvs, torch.nonzero(textures_of_faces == texture).squeeze(-1), width, height
        )
        colours = torch.zeros(len(texels.pixels), 3, device=device)
        weights = torch.zeros(len(texels.pixels), device=device)
        coverage = torch.zeros(len(texels.pixels), device=device)
        for view in views:
            if view.head_only and texture == BODY_TEXTURE:
                continue
            drawn = head_faces if view.head_only else slice(None)
            vertices = posing.vertices(view.video, view.index)
            colour, weight, cosine = _votes(texels, vertices, faces, faces[drawn], view)
            if correction is not None:
                colour = correction.to_reference(view.video, colour)
            colours += weight.unsqueeze(-1) * colour
            weights += weight
            coverage = torch.maximum(coverage, cosine)
        colour_sums = texels.image(colours)
        filled = fill_holes(colour_sums, texels.image(weights.unsqueeze(-1))[..., 0])
        baked[texture] = (filled, texels.image(coverage.unsqueeze(-1))[..., 0])

    def array(tensor: torch.Tensor) -> np.ndarray:
        return tensor.cpu().numpy()

    return TextureRecord(
        body=array(baked[BODY_TEXTURE][0]),
        head=array(baked[HEAD_TEXTURE][0]),
        body_coverage=array(baked[BODY_TEXTURE][1]),
        head_coverage=array(baked[HEAD_TEXTURE][1]),
    )


def _votes(
    texels: Fragments, vertices: torch.Tensor, faces: torch.Tensor, drawn: torch.Tensor,
    view: View,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # fmt: skip
    """What one view says about the texels: ``(P, 3)`` colours, ``(P,)`` vote weights and the
    ``(P,)`` cosine of the angle it sees them at, zero where it does not see them."""
    camera = view.camera
    points = texels.interpolate(vertices[faces])
    normals = triangle_frames(vertices, faces[texels.faces])[0][..., 2]
    screen = camera.screen(points)
    depth = rasterize(camera.screen(vertices), drawn, camera.width, camera.height).depth_image()

    x, y = screen[:, 0].floor().long(), screen[:, 1].floor().long()
    inside = (x >= 0) & (x < camera.width) & (y >= 0) & (y < camera.height)
    x, y = x.clamp(0, camera.width - 1), y.clamp(0, camera.height - 1)
    eroded = -F.max_pool2d(-view.mask[None, None], MASK_EROSION, 1, MASK_EROSION // 2)[0, 0]
    seen = inside & (screen[:, 2] <= depth[y, x] + DEPTH_TOLERANCE)
    seen &= (eroded[y, x] > 0.5) & (view.region[y, x] > 0.5)

    centre = -camera.view[:3, :3].T @ camera.view[:3, 3]
    cosine = (normals * F.normalize(centre - points, dim=-1)).sum(-1).clamp_min(0) * seen
    pixels_per_metre = camera.K[1, 1] / screen[:, 2].clamp_min(1e-3) / 1000
    weight = cosine**SHARPNESS * pixels_per_metre.square()

    image = view.image.float() / 255
    uv = torch.stack((screen[:, 0] / camera.width, 1 - screen[:, 1] / camera.height), dim=-1)
    return sample_texture(image, uv), weight, cosine


def fill_holes(colour_sums: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """``(H, W, 3)`` colours from weighted ``(H, W, 3)`` sums and ``(H, W)`` weights, with the
    texels of zero weight filled from their surroundings (push-pull: averaged down a pyramid, then
    filled in from the coarsest level that has a value)."""
    levels = [(colour_sums.permute(2, 0, 1)[None], weights[None, None])]
    while min(levels[-1][1].shape[-2:]) > 1:
        sums, counts = levels[-1]
        levels.append(
            (F.avg_pool2d(sums, 2, ceil_mode=True), F.avg_pool2d(counts, 2, ceil_mode=True))
        )
    sums, counts = levels[-1]
    filled = sums / counts.clamp_min(1e-12)
    for sums, counts in reversed(levels[:-1]):
        coarse = F.interpolate(filled, sums.shape[-2:], mode="bilinear", align_corners=False)
        filled = torch.where(counts > 0, sums / counts.clamp_min(1e-12), coarse)
    return filled[0].permute(1, 2, 0).clamp(0, 1)


class Textures(torch.nn.Module):
    """The avatar's baked textures and their coverage, on the device that draws them."""

    def __init__(self, record: TextureRecord):
        super().__init__()
        for name in ("body", "head", "body_coverage", "head_coverage"):
            self.register_buffer(name, torch.from_numpy(getattr(record, name)).float())

    def update(self, record: TextureRecord) -> None:
        """Replace the textures and coverage with newly baked ones."""
        for name in ("body", "head", "body_coverage", "head_coverage"):
            getattr(self, name).copy_(torch.from_numpy(getattr(record, name)))

    def textures(self) -> list[torch.Tensor]:
        """The textures in the order of ``Avatar.face_textures``."""
        return [self.body, self.head]

    def coverage(self, uvs: torch.Tensor, textures_of_faces: torch.Tensor) -> torch.Tensor:
        """``(F,)`` coverage at the centre of each triangle."""
        centres = uvs.mean(dim=1)
        coverage = torch.zeros(len(uvs), device=uvs.device)
        maps = {BODY_TEXTURE: self.body_coverage, HEAD_TEXTURE: self.head_coverage}
        for texture, seen in maps.items():
            using = textures_of_faces == texture
            coverage[using] = sample_texture(seen.unsqueeze(-1), centres[using])[:, 0]
        return coverage

    def record(self) -> TextureRecord:
        return TextureRecord(
            body=self.body.cpu().numpy(),
            head=self.head.cpu().numpy(),
            body_coverage=self.body_coverage.cpu().numpy(),
            head_coverage=self.head_coverage.cpu().numpy(),
        )
