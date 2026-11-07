"""Training views: a frame of a video with the avatar posed as the fits found it."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from avatar_splats.body_models.anny import AnnyBody
from avatar_splats.body_models.hybrid import BODY, Avatar, Rig
from avatar_splats.body_models.ict import IctFaceKit
from avatar_splats.cameras import Camera
from avatar_splats.data import read_frames
from avatar_splats.fitting.body import BodyFit, fitted_body
from avatar_splats.fitting.head import CHIN, HeadFit, head_vertices
from avatar_splats.preprocess import Preprocessed


@dataclass(frozen=True)
class View:
    video: str
    frame: int
    image: torch.Tensor
    """``(H, W, 3)`` uint8."""
    mask: torch.Tensor
    """``(H, W)`` float silhouette."""
    region: torch.Tensor
    """``(H, W)`` float: where the image is compared with the render."""
    camera: Camera
    vertices: torch.Tensor
    """``(V, 3)`` the avatar's vertices in this frame."""
    head_only: bool
    """Head videos see the head only; the body's Gaussians stay out of their renders."""


def body_views(
    path, preprocessed: Preprocessed, fit: BodyFit, rig: Rig, body: AnnyBody, resolution: int
) -> list[View]:
    """Views of the body video, with the avatar posed by the body fit."""
    device = rig.rest_vertices.device
    with torch.no_grad():
        vertices = rig(fitted_body(fit, body)[0])
    camera = _scaled_camera(fit.camera(), resolution).to(device)
    images, masks = _load(path, preprocessed, camera, device)
    everywhere = torch.ones_like(masks[0])
    return [
        View("body", int(f), images[i], masks[i], everywhere, camera, vertices[i], head_only=False)
        for i, f in enumerate(preprocessed.frame_indices)
    ]


def head_views(
    video: str,
    path,
    preprocessed: Preprocessed,
    fit: HeadFit,
    avatar: Avatar,
    head: IctFaceKit,
    resolution: int,
) -> list[View]:
    """Views of a head video, with the head posed by its fit; compared above the chin only."""
    device = head.neutral.device

    def tensor(array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(array).float().to(device)

    with torch.no_grad():
        heads = head_vertices(
            head, tensor(fit.identity)[None], tensor(fit.expressions), tensor(fit.rotations),
            tensor(fit.translations), tensor(fit.centre),
        )  # fmt: skip
    body = torch.zeros(avatar.head_start, 3, device=device)
    camera = _scaled_camera(fit.camera(), resolution).to(device)
    chins = camera.project(heads[:, head.landmarks[CHIN]])[:, 1]
    rows = torch.arange(camera.height, device=device, dtype=torch.float32)
    images, masks = _load(path, preprocessed, camera, device)
    return [
        View(
            video,
            int(f),
            images[i],
            masks[i],
            (rows < chins[i]).float()[:, None].expand(-1, camera.width).contiguous(),
            camera,
            torch.cat((body, heads[i])),
            head_only=True,
        )
        for i, f in enumerate(preprocessed.frame_indices)
    ]


def head_gaussians(avatar: Avatar, faces: torch.Tensor) -> torch.Tensor:
    """``(G,)`` bool: which of the Gaussians bound to ``faces`` are on the head."""
    parts = torch.from_numpy(avatar.face_parts).to(faces.device)
    return parts[faces] != BODY


def _scaled_camera(camera: Camera, resolution: int) -> Camera:
    return camera.scaled(min(1.0, resolution / max(camera.width, camera.height)))


def _load(
    path, preprocessed: Preprocessed, camera: Camera, device
) -> tuple[torch.Tensor, torch.Tensor]:
    frames = np.stack(list(read_frames(path)))[preprocessed.frame_indices]
    size = (camera.height, camera.width)
    images = torch.from_numpy(frames).to(device).permute(0, 3, 1, 2).float()
    images = F.interpolate(images, size, mode="area").round().byte().permute(0, 2, 3, 1)
    masks = torch.from_numpy(preprocessed.masks).to(device).float()
    masks = F.interpolate(masks[:, None], size, mode="area")[:, 0]
    return images, masks
