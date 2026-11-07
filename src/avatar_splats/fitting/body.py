"""Fitting Anny to the body turntable: one shape, one pose with small per-frame head movements,
a turntable angle per frame, and per-vertex offsets for what the shape space cannot express."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import roma
import torch
import torch.nn.functional as F

from avatar_splats.body_models.anny import AnnyBody
from avatar_splats.body_models.mesh import uniform_laplacian
from avatar_splats.body_models.skinning import linear_blend_skinning, sparsify_weights
from avatar_splats.cameras import Camera, body_yaw
from avatar_splats.config import FittingConfig
from avatar_splats.fitting.losses import keypoint_loss, second_difference, silhouette_loss
from avatar_splats.fitting.silhouette import SurfaceSamples, render_silhouette, silhouette_iou
from avatar_splats.preprocess import Preprocessed
from avatar_splats.records import ArrayRecord

logger = logging.getLogger(__name__)

POSED_BONES = tuple(
    f"{bone}.{side}" if side else bone
    for bone, sides in {
        "spine05": "", "spine04": "", "spine03": "", "spine02": "", "spine01": "",
        "neck01": "", "neck02": "", "neck03": "", "head": "",
        "clavicle": "LR", "shoulder01": "LR", "upperarm01": "LR", "upperarm02": "LR",
        "lowerarm01": "LR", "lowerarm02": "LR", "wrist": "LR",
        "upperleg01": "LR", "upperleg02": "LR", "lowerleg01": "LR", "lowerleg02": "LR",
        "foot": "LR",
    }.items()
    for side in (sides or [""])
)  # fmt: skip
"""Bones whose rotation is fitted; fingers, toes and the rest keep Anny's rest pose."""

MOVING_BONES = ("neck01", "neck02", "neck03", "head")
"""Bones that may move from frame to frame: the performer holds still but for the head."""

MIN_KEYPOINT_CONFIDENCE = 0.3
BONE_INFLUENCES = 8
"""Bones per vertex when skinning Anny during fitting; more change nothing visible."""


@dataclass(frozen=True)
class BodyFit(ArrayRecord):
    """Anny's parameters for every frame of the body video."""

    frame_indices: np.ndarray
    """``(N,)`` the video frames."""
    phenotype: np.ndarray
    """``(P,)`` Anny phenotype values in [0, 1]."""
    rotations: np.ndarray
    """``(N, J, 3)`` rotation vectors per bone and frame, root first."""
    translation: np.ndarray
    """``(N, 3)`` root translation per frame, in metres."""
    offsets: np.ndarray
    """``(V, 3)`` rest-pose offsets added to Anny's vertices, in metres."""
    fov_y_degrees: float
    image_size: np.ndarray
    """``(2,)`` width and height of the video."""
    silhouette_iou: np.ndarray
    """``(N,)`` IoU of the fitted silhouette with the mask, per frame."""
    keypoint_error: np.ndarray
    """``(N,)`` mean distance in pixels from the fitted to the detected keypoints, per frame."""

    def camera(self) -> Camera:
        width, height = self.image_size
        return Camera.fixed(int(width), int(height), self.fov_y_degrees)


class _BodyParameters(torch.nn.Module):
    def __init__(self, body: AnnyBody, yaw: np.ndarray, translation: torch.Tensor):
        super().__init__()
        frames, bones = len(yaw), body.num_bones
        self.yaw = torch.nn.Parameter(torch.from_numpy(yaw).float())
        self.tilt = torch.nn.Parameter(torch.zeros(3))
        self.translation = torch.nn.Parameter(translation.float())
        self.translation_residual = torch.nn.Parameter(torch.zeros(frames, 3))
        self.phenotype_logit = torch.nn.Parameter(torch.zeros(len(body.phenotype_labels)))
        self.pose = torch.nn.Parameter(torch.zeros(bones, 3))
        self.movement = torch.nn.Parameter(torch.zeros(frames, len(MOVING_BONES), 3))
        self.offsets = torch.nn.Parameter(torch.zeros(body.faces.max().item() + 1, 3))
        self.register_buffer("posed", torch.zeros(bones, 1))
        self.posed[[body.bone(name) for name in POSED_BONES]] = 1
        self.moving = [body.bone(name) for name in MOVING_BONES]

    def phenotype(self) -> torch.Tensor:
        return torch.sigmoid(self.phenotype_logit)

    def rotations(self) -> torch.Tensor:
        frames = len(self.yaw)
        rotations = (self.pose * self.posed).expand(frames, -1, -1).clone()
        rotations[:, self.moving] += self.movement
        yaw = roma.rotvec_to_rotmat(self.yaw.unsqueeze(-1) * self.yaw.new_tensor([0, 0, 1]))
        rotations[:, 0] = roma.rotmat_to_rotvec(yaw @ roma.rotvec_to_rotmat(self.tilt))
        return rotations

    def translations(self) -> torch.Tensor:
        return self.translation + self.translation_residual


def pose_body(
    body: AnnyBody,
    phenotype: torch.Tensor,
    rotations: torch.Tensor,
    translation: torch.Tensor,
    offsets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Anny with per-vertex offsets, posed.

    Args:
        phenotype: ``(1, P)`` shared by all frames.
        rotations: ``(N, J, 3)`` rotation vectors.
        translation: ``(N, 3)`` root translations.
        offsets: ``(V, 3)`` rest-pose vertex offsets.

    Returns:
        ``(N, J, 4, 4)`` skinning matrices and ``(N, V, 3)`` vertices.
    """
    out = body(phenotype, rotations, translation)
    indices, weights = sparsify_weights(body.skin_weights, BONE_INFLUENCES)
    rest = out["rest_vertices"] + offsets
    return out["skinning"], linear_blend_skinning(rest, indices, weights, out["skinning"])


def fitted_body(
    fit: BodyFit, body: AnnyBody, frames: slice | list[int] = slice(None)
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(N, J, 4, 4)`` skinning matrices and ``(N, V, 3)`` vertices of the fit's frames."""
    device = body.faces.device

    def tensor(array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(array).float().to(device)

    return pose_body(
        body,
        tensor(fit.phenotype)[None],
        tensor(fit.rotations[frames]),
        tensor(fit.translation[frames]),
        tensor(fit.offsets),
    )


def fit_body(
    preprocessed: Preprocessed,
    image_size: tuple[int, int],
    body: AnnyBody,
    config: FittingConfig,
) -> BodyFit:
    """Fit Anny to the masks and keypoints of the body video."""
    device = config.device
    width, height = image_size
    scale = config.resolution / max(width, height)
    camera = Camera.fixed(width, height, config.fov_y_degrees).scaled(scale).to(device)

    masks = torch.from_numpy(preprocessed.masks).to(device).float()
    masks = F.interpolate(masks[:, None], (camera.height, camera.width), mode="area")[:, 0]
    keypoints = torch.from_numpy(np.nan_to_num(preprocessed.keypoints)).float().to(device)
    targets = keypoints[..., :2] * scale
    confidence = keypoints[..., 2] * (keypoints[..., 2] >= MIN_KEYPOINT_CONFIDENCE)

    x0, y0, x1, y1 = preprocessed.boxes[0].tolist()
    depth = camera.depth_for_height(config.body_height, (y1 - y0) * scale)
    centre = torch.tensor([(x0 + x1) / 2, (y0 + y1) / 2], device=device) * scale
    start = camera.unproject(centre, depth)
    params = _BodyParameters(body, body_yaw(preprocessed.keypoints), start).to(device)

    rest = body(params.phenotype()[None], params.rotations()[:1], params.translations()[:1])
    samples = SurfaceSamples.draw(
        rest["rest_vertices"][0].detach(), body.faces, config.surface_samples
    )
    radius = 0.8 * samples.spacing
    laplacian = uniform_laplacian(body.faces, len(params.offsets))

    def pose() -> torch.Tensor:
        phenotype, rotations = params.phenotype()[None], params.rotations()
        return pose_body(body, phenotype, rotations, params.translations(), params.offsets)[1]

    def projected_keypoints(vertices: torch.Tensor) -> torch.Tensor:
        return camera.project(body.keypoint_regressor({"vertices": vertices}))

    def height() -> torch.Tensor:
        """Height of the rest pose: a tall body far away and a short one close by look the same,
        so the configured height sets the scale."""
        bones = torch.zeros(1, body.num_bones, 3, device=device)
        rest = body(params.phenotype()[None], bones, bones[:, 0])["rest_vertices"][0]
        z = (rest + params.offsets)[:, 2]
        return z.max() - z.min()

    def loss(phase: int) -> dict[str, torch.Tensor]:
        vertices = pose()
        terms = {
            "height": config.height_weight * (height() / config.body_height - 1).square(),
            "keypoints": config.keypoint_weight
            * keypoint_loss(projected_keypoints(vertices), targets, confidence, camera.height),
            "smoothness": config.smoothness_weight
            * (
                second_difference(params.yaw).square().mean()
                + params.translation_residual.diff(dim=0).square().sum(-1).mean()
                + params.movement.diff(dim=0).square().sum(-1).mean()
            ),
            "pose_prior": config.pose_prior_weight
            * (
                params.pose.square().sum()
                + params.movement.square().sum(-1).mean()
                + 100 * params.translation_residual.square().sum(-1).mean()
            ),
        }
        if phase >= 1:
            silhouettes = render_silhouette(samples.points(vertices, body.faces), radius, camera)
            terms["silhouette"] = config.silhouette_weight * silhouette_loss(silhouettes, masks)
        if phase >= 2:
            terms["detail"] = config.detail_weight * (
                (laplacian @ params.offsets).square().mean() + 0.1 * params.offsets.square().mean()
            )
        return terms

    placement = [params.yaw, params.tilt, params.translation, params.translation_residual]
    placement.append(params.phenotype_logit)
    phases = (
        [{"params": placement}],
        [{"params": placement + [params.pose, params.movement]}],
        [
            {"params": placement + [params.pose, params.movement]},
            {"params": [params.offsets], "lr": config.learning_rate * 0.05},
        ],
    )
    for phase, (iterations, groups) in enumerate(zip(config.iterations, phases, strict=True)):
        optimizer = torch.optim.Adam(groups, lr=config.learning_rate)
        for iteration in range(iterations):
            terms = loss(phase)
            optimizer.zero_grad()
            sum(terms.values()).backward()
            optimizer.step()
            if iteration % 100 == 0 or iteration == iterations - 1:
                logger.info(
                    "body phase %d, iteration %d: %s",
                    phase, iteration, ", ".join(f"{k} {v.item():.2e}" for k, v in terms.items()),
                )  # fmt: skip

    with torch.no_grad():
        vertices = pose()
        silhouettes = render_silhouette(samples.points(vertices, body.faces), radius, camera)
        distance = (projected_keypoints(vertices) - targets).norm(dim=-1) / scale
        seen = confidence > 0
        error = (distance * seen).sum(-1) / seen.sum(-1).clamp_min(1)
        return BodyFit(
            frame_indices=preprocessed.frame_indices,
            phenotype=params.phenotype().cpu().numpy(),
            rotations=params.rotations().cpu().numpy(),
            translation=params.translations().cpu().numpy(),
            offsets=params.offsets.cpu().numpy(),
            fov_y_degrees=config.fov_y_degrees,
            image_size=np.array(image_size),
            silhouette_iou=silhouette_iou(silhouettes, masks).cpu().numpy(),
            keypoint_error=error.cpu().numpy(),
        )
