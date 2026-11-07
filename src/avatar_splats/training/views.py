"""Training views: a frame of a video, its camera, and where to compare it with the render."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from avatar_splats.body_models.hybrid import HEAD_PARTS, Avatar
from avatar_splats.body_models.ict import IctFaceKit
from avatar_splats.cameras import Camera
from avatar_splats.data import read_frames
from avatar_splats.fitting.body import BodyFit
from avatar_splats.fitting.head import CHIN, HeadFit, head_vertices
from avatar_splats.preprocess import Preprocessed


@dataclass(frozen=True)
class View:
    video: str
    index: int
    """Position of the frame among the video's processed frames."""
    frame: int
    """Index of the frame in the video."""
    image: torch.Tensor
    """``(H, W, 3)`` uint8."""
    mask: torch.Tensor
    """``(H, W)`` float silhouette."""
    region: torch.Tensor
    """``(H, W)`` float: where the image is compared with the render."""
    camera: Camera

    @property
    def head_only(self) -> bool:
        """Head videos see the head only; the body's Gaussians stay out of their renders."""
        return self.video != "body"


def body_views(
    path, preprocessed: Preprocessed, fit: BodyFit, resolution: int, device
) -> list[View]:
    """Views of the body video, compared everywhere."""
    camera = _scaled_camera(fit.camera(), resolution).to(device)
    images, masks = _load(path, preprocessed, camera, device)
    everywhere = torch.ones_like(masks[0])
    return [
        View("body", i, int(f), images[i], masks[i], everywhere, camera)
        for i, f in enumerate(preprocessed.frame_indices)
    ]


def head_views(
    video: str,
    path,
    preprocessed: Preprocessed,
    fit: HeadFit,
    head: IctFaceKit,
    resolution: int,
) -> list[View]:
    """Views of a head video, compared above the fitted chin only: below it, the video shows a
    neck and shoulders the head model does not have."""
    device = head.neutral.device

    def tensor(array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(array).float().to(device)

    with torch.no_grad():
        heads = head_vertices(
            head, tensor(fit.identity)[None], tensor(fit.expressions), tensor(fit.rotations),
            tensor(fit.translations), tensor(fit.centre),
        )  # fmt: skip
    camera = _scaled_camera(fit.camera(), resolution).to(device)
    chins = camera.project(heads[:, head.landmarks[CHIN]])[:, 1]
    rows = torch.arange(camera.height, device=device, dtype=torch.float32)
    images, masks = _load(path, preprocessed, camera, device)
    return [
        View(
            video,
            i,
            int(f),
            images[i],
            masks[i],
            (rows < chins[i]).float()[:, None].expand(-1, camera.width).contiguous(),
            camera,
        )
        for i, f in enumerate(preprocessed.frame_indices)
    ]


def head_gaussians(avatar: Avatar, faces: torch.Tensor) -> torch.Tensor:
    """``(G,)`` bool: which of the Gaussians bound to ``faces`` are on the head."""
    parts = torch.from_numpy(avatar.face_parts).to(faces.device)
    return torch.isin(parts[faces], torch.tensor(HEAD_PARTS, device=faces.device))


def _scaled_camera(camera: Camera, resolution: int) -> Camera:
    return camera.scaled(min(1.0, resolution / max(camera.width, camera.height)))


def _load(
    path, preprocessed: Preprocessed, camera: Camera, device, chunk: int = 8
) -> tuple[torch.Tensor, torch.Tensor]:
    """Frames as ``(N, H, W, 3)`` uint8 and masks as ``(N, H, W)`` float, at the camera's size.

    Frames are resized a few at a time: as floats, a whole video would not fit the GPU.
    """
    frames = np.stack(list(read_frames(path)))[preprocessed.frame_indices]
    size = (camera.height, camera.width)
    images, masks = [], []
    for start in range(0, len(frames), chunk):
        batch = torch.from_numpy(frames[start : start + chunk]).to(device)
        batch = F.interpolate(batch.permute(0, 3, 1, 2).float(), size, mode="area")
        images.append(batch.round().byte().permute(0, 2, 3, 1))
        mask = torch.from_numpy(preprocessed.masks[start : start + chunk]).to(device).float()
        masks.append(F.interpolate(mask[:, None], size, mode="area")[:, 0])
    return torch.cat(images), torch.cat(masks)
