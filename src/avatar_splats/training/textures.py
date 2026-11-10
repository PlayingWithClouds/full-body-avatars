"""The avatar's textures, baked from the frames: before training, and again as it goes, from the
poses training has corrected.

Each texel is a weighted average of many frames, so a texture baked this way cannot drift into
noise the way one learned texel by texel does: a texel is seen in few frames, and an optimizer
moves the rarely seen ones by as much as the often seen ones.

Baking projects every texel's point on the surface into every frame. Where a frame sees it, not
hidden by other parts of the mesh, inside the performer's mask and facing the camera, the frame
votes for its colour, the more the more squarely and the more closely it sees it. A texel's
coverage is how squarely the best frame saw it: 1 face on, 0 never. Places no frame sees (the
armpits, the soles) are filled from their surroundings, so they show plausible skin; the canals'
walls, which no frame sees either, take a mucosa colour.

Each frame is split in two bands, baked apart and added up. The broad colour is averaged over
many frames, so neighbouring texels agree and frames of slightly different brightness leave no
patches. The detail, everything up to creases and folds a centimetre or so wide, comes from one
frame per triangle: averaged over several, frames that line up a millimetre apart blur it away,
and where the mesh's shape is wrong (the ears), each frame lands its copy of an ear in a
different place. Each triangle takes the frame that sees it
most squarely and closely, unless its neighbours agree on another one nearly as good, so the
detail comes in patches of one frame each; their seams hide in the broad colour, which is
averaged across them.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from avatar_splats.body_models.hybrid import (
    ATLAS_TILES,
    BODY_TEXTURE,
    CANALS,
    HEAD_PARTS,
    HEAD_TEXTURE,
    Avatar,
)
from avatar_splats.body_models.mesh import adjacent_faces, triangle_frames
from avatar_splats.config import TrainingConfig
from avatar_splats.records import ArrayRecord
from avatar_splats.rendering import Fragments, rasterize, sample_texture
from avatar_splats.training.colour import ColourCorrection
from avatar_splats.training.posing import Posing
from avatar_splats.training.views import View

DEPTH_TOLERANCE = 0.01
"""How far, in metres, a texel's point may lie behind the mesh's depth and still count as seen."""
BROAD_SHARPNESS = 2
"""Power of the cosine of the viewing angle in a frame's vote for the broad colour."""
SEAM_ROUNDS = 10
"""Rounds of letting each triangle reconsider its frame given its neighbours' choices."""
DETAIL_MIN_COSINE = 0.5
"""Frames that see a place more obliquely give it no detail: they smear the frame's pixels along
the surface into streaks."""
MUCOSA = (0.72, 0.4, 0.4)
"""Colour of the canals' walls, which no video sees, unless one does."""
PRIOR_WEIGHT = 1e-9
"""Weight of that colour: less than any frame's vote."""
MASK_EROSION = 5
"""Pixels this close to the mask's edge, where colours mix with the background, have no vote."""


@dataclass(frozen=True)
class TextureRecord(ArrayRecord):
    body: np.ndarray
    """``(H, W, 3)`` float colours in [0, 1] in Anny's UV layout."""
    head: np.ndarray
    """``(H, ATLAS_TILES H, 3)`` float colours in [0, 1] in the atlas of the head and genitals."""
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
    config: TrainingConfig,
    correction: ColourCorrection | None = None,
    warps: dict[tuple[str, int], torch.Tensor] | None = None,
) -> TextureRecord:
    """Bake the avatar's textures from the views: the body's from the body video, the head's
    from all three, the ``config.fallback_videos`` only where the others see nothing.

    Args:
        correction: turns each video's colours into the reference video's before they are
            averaged; without it, the videos' colours are taken as they are.
        warps: by video and frame index, ``(H, W, 2)`` how far, in pixels, each point of the
            posed mesh drawn in that view lies from where the frame shows it (see ``flow``).
    """
    device = posing.rig.faces.device
    faces = posing.rig.faces
    uvs = torch.from_numpy(avatar.uvs).to(device)
    textures_of_faces = torch.from_numpy(avatar.face_textures).to(device)
    parts = torch.from_numpy(avatar.face_parts).to(device)
    on_head = torch.isin(parts, torch.tensor(HEAD_PARTS, device=device))
    head_faces = torch.nonzero(on_head).squeeze(-1)
    body_size, head_size = config.body_texture_size, config.head_texture_size

    def votes(texels: Fragments, view: View) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        drawn = head_faces if view.head_only else slice(None)
        image = view.image.float() / 255
        if correction is not None:
            image = correction.to_reference(view.video, image)
        warp = None if warps is None else warps.get((view.video, view.index))
        vertices = posing.vertices(view.video, view.index)
        bands = torch.cat(_bands(image, _eroded(view), config.detail_scale), -1)
        colour, closeness, cosine = _votes(texels, vertices, faces, faces[drawn], view, bands, warp)
        if view.head_only:  # the head videos pose the head alone
            cosine = cosine * on_head[texels.faces]
        return colour, closeness, cosine

    baked = {}
    for texture, (width, height) in (
        (BODY_TEXTURE, (body_size, body_size)),
        (HEAD_TEXTURE, (ATLAS_TILES * head_size, head_size)),
    ):
        texture_faces = torch.nonzero(textures_of_faces == texture).squeeze(-1)
        texels = texel_fragments(uvs, texture_faces, width, height)
        local = torch.full((len(faces),), -1, dtype=torch.long, device=device)
        local[texture_faces] = torch.arange(len(texture_faces), device=device)
        texel_faces = local[texels.faces]
        texels_per_face = torch.bincount(texel_faces, minlength=len(texture_faces)).clamp_min(1)
        seeing = [v for v in views if not (v.head_only and texture == BODY_TEXTURE)]

        count = len(texels.pixels)
        sums = {tier: torch.zeros(count, 3, device=device) for tier in ("main", "fallback")}
        weights = {tier: torch.zeros(count, device=device) for tier in ("main", "fallback")}
        coverage = torch.zeros(count, device=device)
        quality = torch.zeros(len(texture_faces), len(seeing), device=device)
        for column, view in enumerate(seeing):
            colour, closeness, cosine = votes(texels, view)
            tier = "fallback" if view.video in config.fallback_videos else "main"
            weight = cosine**BROAD_SHARPNESS * closeness
            sums[tier] += weight.unsqueeze(-1) * colour[:, :3]
            weights[tier] += weight
            coverage = torch.maximum(coverage, cosine)
            sharp = cosine * (cosine >= DETAIL_MIN_COSINE)
            quality[:, column].index_add_(0, texel_faces, sharp * closeness)
        canal = (parts[texels.faces] == CANALS).float()
        sums["main"] += canal[:, None] * PRIOR_WEIGHT * torch.tensor(MUCOSA, device=device)
        weights["main"] += canal * PRIOR_WEIGHT
        unseen = weights["main"] == 0
        broad_sums = torch.where(unseen[:, None], sums["fallback"], sums["main"])
        broad_weights = torch.where(unseen, weights["fallback"], weights["main"])
        broad = fill_holes(texels.image(broad_sums), texels.image(broad_weights[:, None])[..., 0])

        fallback = torch.tensor([v.video in config.fallback_videos for v in seeing], device=device)
        chosen = choose_views(
            quality / texels_per_face[:, None],
            fallback,
            adjacent_faces(faces[texture_faces]),
            config.seam_weight,
        )
        detail = torch.zeros(count, 3, device=device)
        for column in torch.unique(chosen[chosen >= 0]).tolist():
            colour, _, cosine = votes(texels, seeing[column])
            taken = (chosen[texel_faces] == column) & (cosine >= DETAIL_MIN_COSINE)
            detail[taken] = colour[taken, 3:]
        detail = texels.image(detail)
        baked[texture] = ((broad + detail).clamp(0, 1), texels.image(coverage[:, None])[..., 0])

    def array(tensor: torch.Tensor) -> np.ndarray:
        return tensor.cpu().numpy()

    return TextureRecord(
        body=array(baked[BODY_TEXTURE][0]),
        head=array(baked[HEAD_TEXTURE][0]),
        body_coverage=array(baked[BODY_TEXTURE][1]),
        head_coverage=array(baked[HEAD_TEXTURE][1]),
    )


def _eroded(view: View) -> torch.Tensor:
    """``(H, W)`` the view's mask without the pixels near its edge, where colours mix with the
    background."""
    mask = view.mask[None, None]
    return -F.max_pool2d(-mask, MASK_EROSION, 1, MASK_EROSION // 2)[0, 0]


def _bands(
    image: torch.Tensor, mask: torch.Tensor, sigma: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(H, W, 3)`` broad colour and fine detail of an ``(H, W, 3)`` image, adding up to it
    inside the ``(H, W)`` mask, split by a blur of standard deviation ``sigma`` pixels. The
    blur averages only pixels inside the mask, so the background does not bleed into the
    performer's edges."""
    radius = math.ceil(3 * sigma)
    offsets = torch.arange(-radius, radius + 1, device=image.device, dtype=image.dtype)
    kernel = torch.exp(-0.5 * (offsets / sigma).square())
    kernel = kernel / kernel.sum()

    def blur(channels: torch.Tensor) -> torch.Tensor:
        count = channels.shape[1]
        rows = kernel.view(1, 1, -1, 1).expand(count, 1, -1, 1)
        columns = kernel.view(1, 1, 1, -1).expand(count, 1, 1, -1)
        channels = F.conv2d(channels, rows, padding=(radius, 0), groups=count)
        return F.conv2d(channels, columns, padding=(0, radius), groups=count)

    weighted = torch.cat((image * mask.unsqueeze(-1), mask.unsqueeze(-1)), -1)
    blurred = blur(weighted.permute(2, 0, 1)[None])[0].permute(1, 2, 0)
    broad = blurred[..., :3] / blurred[..., 3:].clamp_min(1e-6)
    return broad, image - broad


def _votes(
    texels: Fragments, vertices: torch.Tensor, faces: torch.Tensor, drawn: torch.Tensor,
    view: View, image: torch.Tensor, warp: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # fmt: skip
    """What one view says about the texels: ``(P, C)`` values of the ``(H, W, C)`` image at
    them, ``(P,)`` how closely it sees them (squared pixels per millimetre) and the ``(P,)``
    cosine of the angle it sees them at, zero where it does not see them. An ``(H, W, 2)``
    warp moves where the frame is read, in pixels, from where the texels land."""
    camera = view.camera
    points = texels.interpolate(vertices[faces])
    normals = triangle_frames(vertices, faces[texels.faces])[0][..., 2]
    screen = camera.screen(points)
    depth = rasterize(camera.screen(vertices), drawn, camera.width, camera.height).depth_image()

    def pixels(xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = xy[:, 0].floor().long(), xy[:, 1].floor().long()
        inside = (x >= 0) & (x < camera.width) & (y >= 0) & (y < camera.height)
        return x.clamp(0, camera.width - 1), y.clamp(0, camera.height - 1), inside

    x, y, inside = pixels(screen[:, :2])
    seen = inside & (screen[:, 2] <= depth[y, x] + DEPTH_TOLERANCE)
    read = screen[:, :2] if warp is None else screen[:, :2] + warp[y, x]
    x, y, inside = pixels(read)
    seen &= inside & (_eroded(view)[y, x] > 0.5) & (view.region[y, x] > 0.5)

    centre = -camera.view[:3, :3].T @ camera.view[:3, 3]
    cosine = (normals * F.normalize(centre - points, dim=-1)).sum(-1).clamp_min(0) * seen
    pixels_per_millimetre = camera.K[1, 1] / screen[:, 2].clamp_min(1e-3) / 1000

    uv = torch.stack((read[:, 0] / camera.width, 1 - read[:, 1] / camera.height), dim=-1)
    return sample_texture(image, uv), pixels_per_millimetre.square(), cosine


def choose_views(
    quality: torch.Tensor, fallback: torch.Tensor, adjacent: torch.Tensor, seam_weight: float
) -> torch.Tensor:
    """``(F,)`` for every triangle, the view to take its fine detail from; -1 where none sees it.

    Args:
        quality: ``(F, K)`` how well each of the K views sees each triangle, zero where it does
            not: here the mean over its texels of cosine times closeness, zero where obliquer
            than ``DETAIL_MIN_COSINE``.
        fallback: ``(K,)`` bool, views to use only for triangles no other view sees.
        adjacent: ``(A, 2)`` pairs of triangles that share an edge.
        seam_weight: how much a triangle prefers a view each of its neighbours took, in shares
            of the quality of its best view.
    """
    seen = quality > 0
    allowed = seen & ~(fallback & (seen & ~fallback).any(dim=1, keepdim=True))
    quality = torch.where(allowed, quality, 0)
    score = quality / quality.max(dim=1, keepdim=True).values.clamp_min(1e-12)
    score = torch.where(allowed, score, -torch.inf)
    chosen = score.argmax(dim=1)
    bonus = torch.full((len(adjacent),), seam_weight, device=quality.device)
    for _ in range(SEAM_ROUNDS):
        agreement = torch.zeros_like(score)
        for this, other in ((0, 1), (1, 0)):
            agreement.index_put_((adjacent[:, this], chosen[adjacent[:, other]]), bonus, True)
        chosen = (score + agreement).argmax(dim=1)
    return torch.where(allowed.any(dim=1), chosen, -1)


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
