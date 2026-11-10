"""Lining the frames up with the textured mesh in 2D, by optical flow, so bakes keep fine detail.

The fits pose the mesh to within about a centimetre, and the generated videos are not quite
rigid, so the frames disagree by that much about where each mole or crease is. Averaged, the
detail vanishes; aligning each frame's 3D pose by its pixels does not help, since a slightly
wrong pose already matches the broad shading. Optical flow finds instead, for every pixel, where
the frame shows what the mesh draws there, to a fraction of a pixel and whatever the cause.

So the textures are baked, the textured mesh is drawn in every frame's pose, the flow from that
drawing to the frame says where to read the frame for each texel, and the textures are baked
again reading there. Each round's texture is sharper, so the next round's flow is more exact.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from avatar_splats.body_models.hybrid import HEAD_PARTS, Avatar
from avatar_splats.config import TrainingConfig
from avatar_splats.rendering import MeshImage, render_mesh
from avatar_splats.training.colour import ColourCorrection
from avatar_splats.training.posing import Posing
from avatar_splats.training.textures import TextureRecord, Textures, bake_textures
from avatar_splats.training.views import View

MAX_FLOW = 30.0
"""Longest shift, in pixels, a frame may be read from; longer flows are mistakes."""
FLOW_SMOOTHING = 2.0
"""Standard deviation, in pixels, of the blur that steadies the flow."""
EDGE = 7
"""Pixels this close to the edge of the mesh or the mask get no flow: the drawing's black
background and the frame's backdrop differ there."""


@torch.no_grad()
def flow_bake(
    avatar: Avatar,
    views: list[View],
    posing: Posing,
    config: TrainingConfig,
    rounds: int,
    correction: ColourCorrection | None = None,
) -> tuple[TextureRecord, dict[tuple[str, int], torch.Tensor]]:
    """Bake the textures, then ``rounds`` times measure each frame's flow from the textured mesh
    and bake again along it.

    Returns:
        The last textures, and the flows they were baked with, by video and frame index.
    """
    record = bake_textures(avatar, views, posing, config, correction)
    warps: dict[tuple[str, int], torch.Tensor] = {}
    for _ in range(rounds):
        warps = frame_flows(avatar, views, posing, Textures(record), correction)
        record = bake_textures(avatar, views, posing, config, correction, warps)
    return record, warps


@torch.no_grad()
def frame_flows(
    avatar: Avatar,
    views: list[View],
    posing: Posing,
    textures: Textures,
    correction: ColourCorrection | None = None,
) -> dict[tuple[str, int], torch.Tensor]:
    """By video and frame index, ``(H, W, 2)`` where, in pixels from each point of the textured
    mesh drawn in the view, the frame shows the same thing; zero where that is unknown."""
    device = posing.rig.faces.device
    flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    flows = {}
    for view in views:
        mesh = draw(avatar, posing, textures, view)
        frame = view.image.float() / 255
        if correction is not None:
            frame = correction.to_reference(view.video, frame)
        known = _shrink(mesh.mask * (view.mask > 0.5) * view.region, EDGE)
        drawing, seen = _grey(mesh.image, known), _grey(frame, known)
        shift = torch.from_numpy(flow.calc(drawing, seen, None)).to(device)
        flows[(view.video, view.index)] = _steady(shift, known)
    return flows


@torch.no_grad()
def draw(avatar: Avatar, posing: Posing, textures: Textures, view: View) -> MeshImage:
    """The textured mesh posed for a view, over black; head videos see the head alone."""
    device = posing.rig.faces.device
    faces = posing.rig.faces
    drawn = slice(None)
    if view.head_only:
        on_head = torch.isin(torch.from_numpy(avatar.face_parts), torch.tensor(HEAD_PARTS))
        drawn = torch.nonzero(on_head).squeeze(-1).to(device)
    return render_mesh(
        posing.vertices(view.video, view.index),
        faces[drawn],
        torch.from_numpy(avatar.uvs).float().to(device)[drawn],
        torch.from_numpy(avatar.face_textures).long().to(device)[drawn],
        [texture.to(device) for texture in textures.textures()],
        view.camera,
        torch.zeros(3, device=device),
    )


def _shrink(mask: torch.Tensor, pixels: int) -> torch.Tensor:
    size = 2 * pixels + 1
    return -F.max_pool2d(-mask[None, None].float(), size, 1, pixels)[0, 0]


def _grey(image: torch.Tensor, known: torch.Tensor) -> np.ndarray:
    """``(H, W)`` uint8 brightness of an ``(H, W, 3)`` image, its contrast normalised where
    ``known``, so the drawing and the frame look alike to the flow."""
    grey = image.clamp(0, 1).mean(-1)
    inside = grey[known > 0.5]
    if len(inside) > 1:
        grey = (grey - inside.mean()) / inside.std().clamp_min(1e-3) * 40 + 128
    return grey.clamp(0, 255).to(torch.uint8).cpu().numpy()


def _steady(shift: torch.Tensor, known: torch.Tensor) -> torch.Tensor:
    """The ``(H, W, 2)`` flow kept where ``known`` and no longer than ``MAX_FLOW``, blurred with
    ``FLOW_SMOOTHING`` over the known pixels only."""
    length = shift.norm(dim=-1, keepdim=True)
    trusted = known * (length[..., 0] <= MAX_FLOW)
    radius = int(3 * FLOW_SMOOTHING)
    offsets = torch.arange(-radius, radius + 1, device=shift.device, dtype=shift.dtype)
    kernel = torch.exp(-0.5 * (offsets / FLOW_SMOOTHING).square())
    kernel = (kernel / kernel.sum()).outer(kernel / kernel.sum())[None, None].expand(3, 1, -1, -1)
    channels = torch.cat((shift * trusted[..., None], trusted[..., None]), -1).permute(2, 0, 1)
    blurred = F.conv2d(channels[None], kernel, padding=radius, groups=3)[0].permute(1, 2, 0)
    return blurred[..., :2] / blurred[..., 2:].clamp_min(1e-6) * (trusted[..., None] > 0)
