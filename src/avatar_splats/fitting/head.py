"""Fitting the ICT-FaceKit head to the head videos: one identity shared by both videos, and a
rigid head pose and an expression per frame."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import roma
import torch
import torch.nn.functional as F

from avatar_splats.body_models import ict
from avatar_splats.body_models.ict import IctFaceKit
from avatar_splats.cameras import Camera, head_yaw
from avatar_splats.config import FittingConfig
from avatar_splats.fitting.losses import keypoint_loss, second_difference, silhouette_loss
from avatar_splats.fitting.silhouette import SurfaceSamples, render_silhouette, silhouette_iou
from avatar_splats.preprocess import Preprocessed
from avatar_splats.records import ArrayRecord

logger = logging.getLogger(__name__)

MEDIAPIPE_TO_MULTIPIE = [
    162, 234, 93, 58, 172, 136, 149, 148, 152, 377, 378, 365, 397, 288, 323, 454, 389,
    71, 63, 105, 66, 107, 336, 296, 334, 293, 301,
    168, 197, 5, 4, 75, 97, 2, 326, 305,
    33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380,
    61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181, 78, 82, 13, 312, 308, 317, 14, 87,
]  # fmt: skip
"""The MediaPipe face landmark closest to each of the Multi-PIE 68 landmarks."""
JAWLINE = slice(0, 17)
"""The jawline landmarks follow the face's outline, which slides over the mesh as the head
turns, so they count less."""
JAWLINE_WEIGHT = 0.3
CHIN = 8
LEFT_IRIS, RIGHT_IRIS, NOSE_TIP = 473, 468, 4
"""MediaPipe landmarks used to place the head before fitting."""
ICT_NOSE_TIP = 30

ICT_TO_WORLD = torch.tensor([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])
"""ICT-FaceKit's axes (y up, facing +z) to world axes (z up, facing -y)."""


@dataclass(frozen=True)
class HeadFit(ArrayRecord):
    """The head model's parameters for every frame of one head video.

    A frame's head is ``rotation @ ICT_TO_WORLD @ (vertices - centre) + translation``, with the
    vertices of the head model for ``identity`` and that frame's ``expressions``.
    """

    frame_indices: np.ndarray
    identity: np.ndarray
    """``(I,)`` identity weights, shared by all head videos."""
    expressions: np.ndarray
    """``(N, E)`` expression weights in [0, 1]."""
    rotations: np.ndarray
    """``(N, 3)`` head rotation vectors in world axes."""
    translations: np.ndarray
    """``(N, 3)`` in metres."""
    centre: np.ndarray
    """``(3,)`` the point of the head model the head rotates about, in its own coordinates."""
    fov_y_degrees: float
    image_size: np.ndarray
    silhouette_iou: np.ndarray
    """``(N,)`` IoU above the chin, per frame."""
    landmark_error: np.ndarray
    """``(N,)`` mean landmark distance in pixels; NaN where no face was found."""

    def camera(self) -> Camera:
        width, height = self.image_size
        return Camera.fixed(int(width), int(height), self.fov_y_degrees)


def head_vertices(
    model: IctFaceKit,
    identity: torch.Tensor,
    expressions: torch.Tensor,
    rotations: torch.Tensor,
    translations: torch.Tensor,
    centre: torch.Tensor,
) -> torch.Tensor:
    """``(N, V, 3)`` world vertices of the head for per-frame expressions and poses."""
    local = model(identity, expressions) - centre
    rotation = roma.rotvec_to_rotmat(rotations) @ ICT_TO_WORLD.to(local)
    return local @ rotation.transpose(-1, -2) + translations.unsqueeze(-2)


class _VideoParameters(torch.nn.Module):
    def __init__(self, yaw: np.ndarray, translation: torch.Tensor, num_expressions: int):
        super().__init__()
        frames = len(yaw)
        rotations = torch.zeros(frames, 3)
        rotations[:, 2] = torch.from_numpy(yaw).float()
        self.rotations = torch.nn.Parameter(rotations)
        self.translations = torch.nn.Parameter(translation.float().expand(frames, 3).clone())
        self.expression_logits = torch.nn.Parameter(torch.full((frames, num_expressions), -5.0))

    def expressions(self) -> torch.Tensor:
        return torch.sigmoid(self.expression_logits)


@dataclass
class _Video:
    preprocessed: Preprocessed
    image_size: tuple[int, int]
    camera: Camera
    scale: float
    masks: torch.Tensor
    landmarks: torch.Tensor
    landmark_weights: torch.Tensor
    params: _VideoParameters


def fit_heads(
    videos: dict[str, tuple[Preprocessed, tuple[int, int]]],
    model: IctFaceKit,
    config: FittingConfig,
) -> dict[str, HeadFit]:
    """Fit one head identity and per-frame poses and expressions to several head videos.

    Args:
        videos: per video name, its preprocessing results and its image size.
    """
    device = config.device
    model = model.to(device)
    identity = torch.nn.Parameter(torch.zeros(1, model.num_identities, device=device))
    neutral = model.neutral[ict.SKIN]
    centre = neutral.mean(dim=0)
    skin_faces = model.faces[model.faces.max(dim=1).values < ict.SKIN.stop]
    samples = SurfaceSamples.draw(model.neutral, skin_faces, config.surface_samples)
    radius = 0.8 * samples.spacing
    landmark_vertices = model.landmarks.to(device)

    fits = {
        name: _prepare(pre, size, model, centre, config) for name, (pre, size) in videos.items()
    }

    def pose(video: _Video) -> torch.Tensor:
        p = video.params
        return head_vertices(model, identity, p.expressions(), p.rotations, p.translations, centre)

    def chin_region(video: _Video, vertices: torch.Tensor) -> torch.Tensor:
        chin = video.camera.project(vertices[:, landmark_vertices[CHIN]])[:, 1].detach()
        rows = torch.arange(video.camera.height, device=device).float()
        return (rows[None, :] < chin[:, None]).float()[:, :, None].expand_as(video.masks)

    def loss(phase: int) -> dict[str, torch.Tensor]:
        terms: dict[str, torch.Tensor] = {}
        for name, video in fits.items():
            vertices = pose(video)
            p = video.params
            projected = video.camera.project(vertices[:, landmark_vertices])
            terms[f"{name} landmarks"] = config.keypoint_weight * keypoint_loss(
                projected, video.landmarks, video.landmark_weights, video.camera.height
            )
            terms[f"{name} smoothness"] = config.smoothness_weight * (
                second_difference(p.rotations).square().sum(-1).mean()
                + second_difference(p.translations).square().sum(-1).mean()
                + 0.01 * p.expressions().diff(dim=0).square().sum(-1).mean()
            )
            if phase >= 1:
                rendered = render_silhouette(
                    samples.points(vertices, skin_faces), radius, video.camera
                )
                terms[f"{name} silhouette"] = config.silhouette_weight * silhouette_loss(
                    rendered, video.masks, chin_region(video, vertices)
                )
            if phase >= 2:
                terms[f"{name} sparsity"] = (
                    config.expression_sparsity_weight * p.expressions().sum(-1).mean()
                )
        terms["identity prior"] = config.identity_prior_weight * identity.square().sum()
        return terms

    rigid = [
        p for video in fits.values() for p in (video.params.rotations, video.params.translations)
    ]
    expression = [video.params.expression_logits for video in fits.values()]
    phases = (rigid, rigid + [identity], rigid + [identity] + expression)
    for phase, (iterations, trainable) in enumerate(zip(config.iterations, phases, strict=True)):
        optimizer = torch.optim.Adam(trainable, lr=config.learning_rate)
        for iteration in range(iterations):
            terms = loss(phase)
            optimizer.zero_grad()
            sum(terms.values()).backward()
            optimizer.step()
            if iteration % 100 == 0 or iteration == iterations - 1:
                logger.info(
                    "head phase %d, iteration %d: %s",
                    phase, iteration, ", ".join(f"{k} {v.item():.2e}" for k, v in terms.items()),
                )  # fmt: skip

    results = {}
    with torch.no_grad():
        for name, video in fits.items():
            vertices = pose(video)
            rendered = render_silhouette(samples.points(vertices, skin_faces), radius, video.camera)
            region = chin_region(video, vertices).bool()
            iou = silhouette_iou(rendered * region, video.masks * region)
            projected = video.camera.project(vertices[:, landmark_vertices])
            distance = (projected - video.landmarks).norm(dim=-1).mean(-1) / video.scale
            found = video.landmark_weights.sum(-1) > 0
            p = video.params
            results[name] = HeadFit(
                frame_indices=video.preprocessed.frame_indices,
                identity=identity[0].cpu().numpy(),
                expressions=p.expressions().cpu().numpy(),
                rotations=p.rotations.cpu().numpy(),
                translations=p.translations.cpu().numpy(),
                centre=centre.cpu().numpy(),
                fov_y_degrees=config.fov_y_degrees,
                image_size=np.array(video.image_size),
                silhouette_iou=iou.cpu().numpy(),
                landmark_error=torch.where(found, distance, torch.nan).cpu().numpy(),
            )
    return results


def _prepare(
    preprocessed: Preprocessed,
    image_size: tuple[int, int],
    model: IctFaceKit,
    centre: torch.Tensor,
    config: FittingConfig,
) -> _Video:
    device = config.device
    width, height = image_size
    scale = config.resolution / max(width, height)
    camera = Camera.fixed(width, height, config.fov_y_degrees).scaled(scale).to(device)

    masks = torch.from_numpy(preprocessed.masks).to(device).float()
    masks = F.interpolate(masks[:, None], (camera.height, camera.width), mode="area")[:, 0]

    mediapipe = torch.from_numpy(preprocessed.face_landmarks).float().to(device) * scale
    found = ~mediapipe.isnan().any(dim=-1).any(dim=-1)
    landmarks = torch.nan_to_num(mediapipe[:, MEDIAPIPE_TO_MULTIPIE, :2])
    weights = torch.ones(68, device=device)
    weights[JAWLINE] = JAWLINE_WEIGHT
    weights = weights * found[:, None]

    yaw = head_yaw(preprocessed.face_landmarks)
    first = int(found.float().argmax())
    eyes = mediapipe[first, [LEFT_IRIS, RIGHT_IRIS], :2]
    depth = camera.depth_for_height(config.eye_distance, float((eyes[0] - eyes[1]).norm()))
    nose = camera.unproject(mediapipe[first, NOSE_TIP, :2], depth)
    rotation = roma.rotvec_to_rotmat(torch.tensor([0, 0, float(yaw[first])], device=device))
    nose_offset = (
        rotation @ ICT_TO_WORLD.to(device) @ (model.neutral[model.landmarks[ICT_NOSE_TIP]] - centre)
    )
    params = _VideoParameters(yaw, nose - nose_offset, model.num_expressions).to(device)
    return _Video(preprocessed, image_size, camera, scale, masks, landmarks, weights, params)
