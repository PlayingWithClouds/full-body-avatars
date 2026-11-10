"""Fitting the ICT-FaceKit head to the head videos: one identity shared by both videos, and a
rigid head pose, an expression and a gaze per frame.

ICT-FaceKit's ``eyeLook`` blendshapes move the eyelids, not the eyeballs, so the eyes are turned
apart, to put the pupils on MediaPipe's iris centres. Unturned, every frame paints the iris onto a
different patch of the eyeball, and baked from all of them it turns into a grey smear."""

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
"""MediaPipe landmarks used to place the head before fitting; the irises also aim the eyes."""
ICT_NOSE_TIP = 30

ICT_TO_WORLD = torch.tensor([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])
"""ICT-FaceKit's axes (y up, facing +z) to world axes (z up, facing -y)."""


@dataclass(frozen=True)
class HeadFit(ArrayRecord):
    """The head model's parameters for every frame of one head video.

    A frame's head is ``rotation @ ICT_TO_WORLD @ (vertices - centre) + translation``, with the
    vertices of the head model for ``identity`` and that frame's ``expressions`` and
    ``eye_rotations``.
    """

    frame_indices: np.ndarray
    identity: np.ndarray
    """``(I,)`` identity weights, shared by all head videos."""
    expressions: np.ndarray
    """``(N, E)`` expression weights in [0, 1]."""
    eye_rotations: np.ndarray
    """``(N, 2, 3)`` rotation vectors of the eyeballs about their centres, in the head model's
    axes, in ``ict.EYEBALL_ORDER``."""
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
    eye_rotations: torch.Tensor,
    rotations: torch.Tensor,
    translations: torch.Tensor,
    centre: torch.Tensor,
) -> torch.Tensor:
    """``(N, V, 3)`` world vertices of the head for per-frame expressions, gazes and poses."""
    local = ict.turn_eyes(model(identity, expressions), eye_rotations) - centre
    rotation = roma.rotvec_to_rotmat(rotations) @ ICT_TO_WORLD.to(local)
    return local @ rotation.transpose(-1, -2) + translations.unsqueeze(-2)


class VideoParameters(torch.nn.Module):
    def __init__(self, yaw: np.ndarray, translation: torch.Tensor, num_expressions: int):
        super().__init__()
        frames = len(yaw)
        rotations = torch.zeros(frames, 3)
        rotations[:, 2] = torch.from_numpy(yaw).float()
        self.rotations = torch.nn.Parameter(rotations)
        self.translations = torch.nn.Parameter(translation.float().expand(frames, 3).clone())
        self.expression_logits = torch.nn.Parameter(torch.full((frames, num_expressions), -5.0))
        self.eye_angles = torch.nn.Parameter(torch.zeros(frames, 2, 2))
        """Per eye, the turn up or down (about the head model's x) and sideways (about its y)."""

    def expressions(self) -> torch.Tensor:
        return torch.sigmoid(self.expression_logits)

    def eye_rotations(self) -> torch.Tensor:
        return F.pad(self.eye_angles, (0, 1))


@dataclass
class _Video:
    preprocessed: Preprocessed
    image_size: tuple[int, int]
    camera: Camera
    scale: float
    masks: torch.Tensor
    compared: torch.Tensor
    """``(N, H, W)`` 1 where the silhouette is compared, 0 on ``Preprocessed.ignored``."""
    landmarks: torch.Tensor
    landmark_weights: torch.Tensor
    irises: torch.Tensor
    """``(N, 2, 2)`` iris centres in pixels, in ``ict.EYEBALL_ORDER``."""
    found: torch.Tensor
    """``(N,)`` float: 1 where MediaPipe found the face."""
    params: VideoParameters


def fit_heads(
    videos: dict[str, tuple[Preprocessed, tuple[int, int]]],
    model: IctFaceKit,
    config: FittingConfig,
    outlier_factor: float | None = None,
) -> dict[str, HeadFit]:
    """Fit one head identity and per-frame poses and expressions to several head videos.

    Args:
        videos: per video name, its preprocessing results and its image size.
        outlier_factor: after the first phase (the head placed rigidly on the landmarks),
            faces whose landmarks it misses by more than this many times the median over every
            video are left out of the rest: MediaPipe fits its mesh badly to faces seen from
            below or far turned, and the identity would bend to them. None: all are kept.
    """
    device = config.device
    model = model.to(device)
    identity = torch.nn.Parameter(torch.zeros(1, model.num_identities, device=device))
    neutral = model.neutral[ict.SKIN]
    centre = neutral.mean(dim=0)
    skin_faces = model.faces[model.faces.max(dim=1).values < ict.SKIN.stop]
    samples = SurfaceSamples.draw(model.neutral, skin_faces, config.surface_samples)
    radius = samples.radius
    inset = config.silhouette_inset * radius
    landmark_vertices = model.landmarks.to(device)
    pupils = model.pupils().to(device)

    fits = {
        name: _prepare(pre, size, model, centre, config) for name, (pre, size) in videos.items()
    }

    def pose(video: _Video) -> torch.Tensor:
        p = video.params
        return head_vertices(
            model, identity, p.expressions(), p.eye_rotations(), p.rotations, p.translations, centre
        )

    def chin_region(video: _Video, vertices: torch.Tensor) -> torch.Tensor:
        chin = video.camera.project(vertices[:, landmark_vertices[CHIN]])[:, 1].detach()
        rows = torch.arange(video.camera.height, device=device).float()
        above = (rows[None, :] < chin[:, None]).float()[:, :, None]
        return above * video.compared

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
                    samples.points(vertices, skin_faces, inset), radius, video.camera
                )
                terms[f"{name} silhouette"] = config.silhouette_weight * silhouette_loss(
                    rendered, video.masks, chin_region(video, vertices)
                )
            if phase >= 2:
                terms[f"{name} sparsity"] = (
                    config.expression_sparsity_weight * p.expressions().sum(-1).mean()
                )
                terms[f"{name} irises"] = config.keypoint_weight * keypoint_loss(
                    video.camera.project(vertices[:, pupils]),
                    video.irises,
                    video.found[:, None] * facing(vertices, pupils).detach(),
                    video.camera.height,
                )
                terms[f"{name} gaze"] = (
                    config.gaze_prior_weight * p.eye_angles.square().sum((1, 2)).mean()
                    + config.smoothness_weight
                    * second_difference(p.eye_angles).square().sum((1, 2)).mean()
                )
        terms["identity prior"] = config.identity_prior_weight * identity.square().sum()
        return terms

    rigid = [
        p for video in fits.values() for p in (video.params.rotations, video.params.translations)
    ]
    expression = [
        p
        for video in fits.values()
        for p in (video.params.expression_logits, video.params.eye_angles)
    ]
    phases = (rigid, rigid + [identity], rigid + [identity] + expression)
    for phase, (iterations, trainable) in enumerate(zip(config.iterations, phases, strict=True)):
        if phase == 1 and outlier_factor is not None:
            _drop_outliers(fits, pose, landmark_vertices, outlier_factor)
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
            points = samples.points(vertices, skin_faces, inset)
            rendered = render_silhouette(points, radius, video.camera)
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
                eye_rotations=p.eye_rotations().cpu().numpy(),
                rotations=p.rotations.cpu().numpy(),
                translations=p.translations.cpu().numpy(),
                centre=centre.cpu().numpy(),
                fov_y_degrees=config.fov_y_degrees,
                image_size=np.array(video.image_size),
                silhouette_iou=iou.cpu().numpy(),
                landmark_error=torch.where(found, distance, torch.nan).cpu().numpy(),
            )
    return results


@torch.no_grad()
def _drop_outliers(fits: dict[str, _Video], pose, landmark_vertices, factor: float) -> None:
    """Leave out the faces whose landmarks the current fit misses by more than ``factor``
    times the median miss over all videos' faces (in image heights)."""
    misses = {}
    for name, video in fits.items():
        projected = video.camera.project(pose(video)[:, landmark_vertices])
        distance = (projected - video.landmarks).norm(dim=-1).mean(-1) / video.camera.height
        found = video.landmark_weights.sum(-1) > 0
        misses[name] = torch.where(found, distance, torch.nan)
    median = torch.cat(list(misses.values())).nanmedian()
    for name, video in fits.items():
        bad = misses[name] > factor * median
        if bad.any():
            video.landmark_weights[bad] = 0
            video.found[bad] = 0
            logger.info(
                "head video %s: %d of %d faces left out, their landmarks missed by more than "
                "%.1f times the median", name, int(bad.sum()), int((misses[name] >= 0).sum()),
                factor,
            )  # fmt: skip


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
    compared = torch.ones_like(masks)
    if preprocessed.ignored is not None:
        ignored = torch.from_numpy(preprocessed.ignored).to(device).float()
        ignored = F.interpolate(ignored[:, None], (camera.height, camera.width), mode="area")
        compared = (ignored[:, 0] < 0.5).float()
    targets = landmark_targets(preprocessed.face_landmarks, scale, model, device)
    params = initial_parameters(
        preprocessed.face_landmarks, targets, camera, model, centre, config.eye_distance
    )
    return _Video(
        preprocessed, image_size, camera, scale, masks, compared, targets.landmarks,
        targets.weights, targets.irises, targets.found, params,
    )  # fmt: skip


@dataclass
class LandmarkTargets:
    """What a video's MediaPipe face landmarks ask of the head model, scaled to the fitting
    resolution."""

    mediapipe: torch.Tensor
    """``(N, 478, 3)`` the landmarks in pixels, NaN where no face was found."""
    landmarks: torch.Tensor
    """``(N, 68, 2)`` pixels: MediaPipe's counterparts of the head model's 68 landmarks."""
    weights: torch.Tensor
    """``(N, 68)`` how much each landmark counts; zero throughout where no face was found."""
    irises: torch.Tensor
    """``(N, 2, 2)`` iris centres in pixels, in ``ict.EYEBALL_ORDER``."""
    found: torch.Tensor
    """``(N,)`` float: 1 where MediaPipe found the face."""


def landmark_targets(
    face_landmarks: np.ndarray, scale: float, model: IctFaceKit, device: str
) -> LandmarkTargets:
    """The head model's landmark targets from ``(N, 478, 3)`` MediaPipe landmarks in pixels, for
    images scaled by ``scale``."""
    mediapipe = torch.from_numpy(face_landmarks).float().to(device) * scale
    found = ~mediapipe.isnan().any(dim=-1).any(dim=-1)
    landmarks = torch.nan_to_num(mediapipe[:, MEDIAPIPE_TO_MULTIPIE, :2])
    weights = torch.ones(68, device=device)
    weights[JAWLINE] = JAWLINE_WEIGHT
    weights = weights * found[:, None]

    left_first = (
        model.neutral[ict.LEFT_EYEBALL, 0].mean() > model.neutral[ict.RIGHT_EYEBALL, 0].mean()
    )
    iris_landmarks = [LEFT_IRIS, RIGHT_IRIS] if left_first else [RIGHT_IRIS, LEFT_IRIS]
    irises = torch.nan_to_num(mediapipe[:, iris_landmarks, :2])
    return LandmarkTargets(mediapipe, landmarks, weights, irises, found.float())


def initial_parameters(
    face_landmarks: np.ndarray,
    targets: LandmarkTargets,
    camera: Camera,
    model: IctFaceKit,
    centre: torch.Tensor,
    eye_distance: float,
) -> VideoParameters:
    """Parameters starting from the turntable's yaw per frame, the head at the distance its
    eyes' spacing implies, and a neutral face."""
    mediapipe = targets.mediapipe
    yaw = head_yaw(face_landmarks)
    first = int(targets.found.argmax())
    eyes = mediapipe[first, [LEFT_IRIS, RIGHT_IRIS], :2]
    depth = camera.depth_for_height(eye_distance, float((eyes[0] - eyes[1]).norm()))
    nose = camera.unproject(mediapipe[first, NOSE_TIP, :2], depth)
    rotation = roma.rotvec_to_rotmat(
        torch.tensor([0, 0, float(yaw[first])], device=mediapipe.device)
    )
    nose_offset = (
        rotation
        @ ICT_TO_WORLD.to(mediapipe.device)
        @ (model.neutral[model.landmarks[ICT_NOSE_TIP]] - centre)
    )
    return VideoParameters(yaw, nose - nose_offset, model.num_expressions).to(mediapipe.device)


def facing(vertices: torch.Tensor, pupils: torch.Tensor) -> torch.Tensor:
    """``(N, 2)`` how squarely the camera at the origin sees each eye: the cosine between the
    eyeball's forward direction and the direction to the camera, zero when it looks away.
    MediaPipe guesses an eye it sees from the side or not at all."""
    centres = torch.stack(
        [vertices[:, eyeball].mean(dim=1) for eyeball in ict.EYEBALL_ORDER], dim=1
    )
    forward = F.normalize(vertices[:, pupils] - centres, dim=-1)
    return (forward * F.normalize(-vertices[:, pupils], dim=-1)).sum(-1).clamp_min(0)
