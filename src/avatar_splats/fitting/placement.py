"""Placing the fitted head on the fitted body so that the two agree with the body video.

The head comes from the head videos and the body from the body video; matching the head model to
Anny's head only gets it close. If the avatar's head sits even a few millimetres off where the body
video shows it, training sees two heads and learns both. So the placement is refined against the
body video: its silhouette above the shoulders and the detected nose and eyes, in every frame.
"""

from __future__ import annotations

import logging

import numpy as np
import roma
import torch
import torch.nn.functional as F

from avatar_splats.body_models import ict
from avatar_splats.body_models.anny import AnnyBody
from avatar_splats.body_models.hybrid import (
    ICT_LEFT_EYE,
    ICT_NOSE_TIP,
    ICT_RIGHT_EYE,
    HeadPlacement,
    transfer_weights,
)
from avatar_splats.body_models.ict import IctFaceKit
from avatar_splats.body_models.skinning import linear_blend_skinning, sparsify_weights
from avatar_splats.cameras import Camera
from avatar_splats.config import FittingConfig
from avatar_splats.fitting.body import (
    BONE_INFLUENCES,
    MIN_KEYPOINT_CONFIDENCE,
    BodyFit,
    fitted_body,
)
from avatar_splats.fitting.losses import keypoint_loss, silhouette_loss
from avatar_splats.fitting.silhouette import SurfaceSamples, render_silhouette
from avatar_splats.preprocess import Preprocessed

logger = logging.getLogger(__name__)

COCO_FACE = [0, 1, 2]
"""Nose, left eye, right eye: the COCO keypoints the head model has counterparts for."""
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
ITERATIONS = 300


def place_head(
    placement: HeadPlacement,
    head: IctFaceKit,
    head_rest: torch.Tensor,
    body: AnnyBody,
    body_rest: torch.Tensor,
    fit: BodyFit,
    preprocessed: Preprocessed,
    config: FittingConfig,
) -> HeadPlacement:
    """Refine the head's placement on the body against the body video.

    Args:
        placement: where the head starts, e.g. from ``hybrid.match_heads``.
        head_rest: ``(V, 3)`` the fitted head in its own coordinates.
        body_rest: ``(V, 3)`` the fitted body in its rest pose.
    """
    device = config.device
    width, height = fit.image_size
    scale = config.resolution / max(width, height)
    camera = Camera.fixed(int(width), int(height), config.fov_y_degrees).scaled(scale).to(device)
    masks = torch.from_numpy(preprocessed.masks).to(device).float()
    masks = F.interpolate(masks[:, None], (camera.height, camera.width), mode="area")[:, 0]
    keypoints = torch.from_numpy(np.nan_to_num(preprocessed.keypoints)).float().to(device)
    targets = keypoints[:, COCO_FACE, :2] * scale
    confidence = keypoints[:, COCO_FACE, 2]
    confidence = confidence * (confidence >= MIN_KEYPOINT_CONFIDENCE)

    # Above the shoulders, the silhouette is the head and neck.
    shoulders = keypoints[:, [LEFT_SHOULDER, RIGHT_SHOULDER], 1].mean(dim=1) * scale
    rows = torch.arange(camera.height, device=device, dtype=torch.float32)
    region = (rows[None, :] < shoulders[:, None]).float()[:, :, None].expand_as(masks)

    with torch.no_grad():
        skinning, body_vertices = fitted_body(fit, body)
        start = placement.apply(head_rest)
        bones, weights = sparsify_weights(transfer_weights(body, body_rest, start), BONE_INFLUENCES)
        torso = body.faces[~body.head_faces]
        torso_samples = SurfaceSamples.draw(body_rest, torso, config.surface_samples)
        torso_points = torso_samples.points(body_vertices, torso)
    skin = head.faces[head.faces.max(dim=1).values < ict.SKIN.stop]
    head_samples = SurfaceSamples.draw(start, skin, config.surface_samples // 2)
    radius = 0.8 * min(torso_samples.spacing, head_samples.spacing)
    landmarks = head.landmarks
    centre = start[ict.SKIN].mean(dim=0)

    log_scale = torch.nn.Parameter(torch.zeros((), device=device))
    rotvec = torch.nn.Parameter(torch.zeros(3, device=device))
    shift = torch.nn.Parameter(torch.zeros(3, device=device))

    def adjust(points: torch.Tensor) -> torch.Tensor:
        """The refinement: a similarity transform about the head's centre."""
        turned = (points - centre) @ roma.rotvec_to_rotmat(rotvec).T
        return log_scale.exp() * turned + centre + shift

    def loss() -> dict[str, torch.Tensor]:
        posed = linear_blend_skinning(adjust(start), bones, weights, skinning)
        face = torch.stack(
            (
                posed[:, landmarks[ICT_NOSE_TIP]],
                posed[:, landmarks[list(ICT_LEFT_EYE)]].mean(dim=1),
                posed[:, landmarks[list(ICT_RIGHT_EYE)]].mean(dim=1),
            ),
            dim=1,
        )
        points = torch.cat((torso_points, head_samples.points(posed, skin)), dim=1)
        silhouettes = render_silhouette(points, radius, camera)
        return {
            "keypoints": config.keypoint_weight
            * keypoint_loss(camera.project(face), targets, confidence, camera.height),
            "silhouette": config.silhouette_weight * silhouette_loss(silhouettes, masks, region),
        }

    optimizer = torch.optim.Adam([log_scale, rotvec, shift], lr=1e-3)
    for iteration in range(ITERATIONS):
        terms = loss()
        optimizer.zero_grad()
        sum(terms.values()).backward()
        optimizer.step()
        if iteration % 100 == 0 or iteration == ITERATIONS - 1:
            logger.info(
                "head placement, iteration %d: %s",
                iteration,
                ", ".join(f"{k} {v.item():.2e}" for k, v in terms.items()),
            )

    with torch.no_grad():
        logger.info(
            "head placement refined: moved %.1f mm, turned %.1f degrees, scaled by %.3f",
            1000 * shift.norm().item(),
            np.degrees(rotvec.norm().item()),
            log_scale.exp().item(),
        )
        # Compose the refinement with the starting placement: x -> s R (P(x) - c) + c + t.
        turn = roma.rotvec_to_rotmat(rotvec)
        factor = log_scale.exp().item()
        return HeadPlacement(
            scale=factor * placement.scale,
            rotation=turn @ placement.rotation,
            translation=factor * turn @ (placement.translation - centre) + centre + shift,
        )
