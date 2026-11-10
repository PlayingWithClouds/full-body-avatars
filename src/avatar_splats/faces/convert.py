"""From a video's face to the avatar's face.

The avatar's head is the ICT-FaceKit model with the performer's identity, so its face is the
model's expression weights (the avatar's blendshapes are the model's, scaled and turned onto the
body) and the head's rotation. Two things tell the weights:

- MediaPipe's ARKit blendshapes, calibrated to the performer's neutral face (``blendshapes``),
  are the estimate. They are learned from many faces and see what a few landmarks cannot: a
  grin's stretched lips and bared teeth, a frown, a wrinkled nose.
- All 468 mesh landmarks (``dense``) refine it: the head model, posed by a rigid transform and
  the weights, is laid over them, with a prior holding the weights near the estimate where the
  landmarks say little. The eyeballs' turn is fitted to the irises.

The identity is the performer's, fixed, and nothing but the video's own detections is needed, so
any head video converts, not only the capture's.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from avatar_splats.body_models import ict
from avatar_splats.body_models.ict import IctFaceKit
from avatar_splats.cameras import Camera
from avatar_splats.config import Config
from avatar_splats.faces import dense
from avatar_splats.fitting.head import (
    CHIN,
    ICT_TO_WORLD,
    JAWLINE_WEIGHT,
    LandmarkTargets,
    VideoParameters,
    facing,
    head_vertices,
    initial_parameters,
    landmark_targets,
)
from avatar_splats.fitting.losses import keypoint_loss, second_difference
from avatar_splats.records import ArrayRecord


@dataclass(frozen=True)
class FaceFit(ArrayRecord):
    """The avatar's face in every frame of a video."""

    frame_indices: np.ndarray
    """``(N,)`` int: index of each processed frame in the video."""
    fps: float
    """Frames per second of the processed frames."""
    expression_names: np.ndarray
    """``(E,)`` str: the avatar's expressions."""
    expressions: np.ndarray
    """``(N, E)`` expression weights in [0, 1], in the avatar's order."""
    rotations: np.ndarray
    """``(N, 3)`` rotation vectors of the head in world axes (the camera looks along +y, the
    head faces -y when the rotation is zero), about the head's centre."""
    eye_rotations: np.ndarray
    """``(N, 2, 3)`` rotation vectors of the eyeballs about their centres, in world axes, for
    the eyeball on the avatar's +x side, then the one on its -x side. Zero looks straight ahead
    of the head."""
    found: np.ndarray
    """``(N,)`` bool: whether a face was found. Elsewhere the face is held by its neighbours."""
    estimate: np.ndarray
    """``(N, E)`` the calibrated blendshape estimate the weights were refined from, in the
    avatar's order."""
    landmark_error: np.ndarray
    """``(N,)`` mean distance in pixels of the video between the head's 68 landmarks (the head
    fit's, comparable with ``HeadFit.landmark_error``) and the detected ones; NaN where no face
    was found."""
    estimate_error: np.ndarray
    """``(N,)`` the same with the estimate as the weights, before the refinement."""
    dense_error: np.ndarray
    """``(N,)`` mean distance in pixels over all 468 mesh landmarks."""
    outline: np.ndarray
    """``(N, 4)`` x0, y0, x1, y1 in pixels: where the posed head lies in the frame."""


def match_expressions(
    weights: np.ndarray, names: tuple[str, ...] | list[str], target_names: list[str]
) -> np.ndarray:
    """``(N, len(target_names))`` the ``(N, len(names))`` weights ordered by ``target_names``;
    expressions the weights do not have (the genitals' shapes) stay at zero."""
    index = {name: i for i, name in enumerate(names)}
    out = np.zeros((len(weights), len(target_names)), weights.dtype)
    for target, name in enumerate(target_names):
        if name in index:
            out[:, target] = weights[:, index[name]]
    return out


def convert_faces(
    face_landmarks: np.ndarray,
    frame_indices: np.ndarray,
    image_size: tuple[int, int],
    fps: float,
    model: IctFaceKit,
    identity: np.ndarray,
    expression_names: list[str],
    estimate: np.ndarray,
    mesh_vertices: np.ndarray,
    config: Config,
) -> FaceFit:
    """Convert a video's face landmarks to the avatar's expressions and head turns.

    Args:
        face_landmarks: ``(N, 478, 3)`` MediaPipe landmarks as x, y in pixels and depth; NaN
            where no face was found.
        frame_indices: ``(N,)`` index of each frame in the video.
        image_size: width and height of the video in pixels.
        fps: frames per second of the processed frames.
        identity: ``(I,)`` the performer's identity weights.
        expression_names: the avatar's expressions.
        estimate: ``(N, E_model)`` weights in [0, 1] of the head model's expressions, from
            the calibrated blendshapes, that the fit starts from and is held near.
        mesh_vertices: ``(468,)`` the head model's skin vertex under each mesh landmark
            (``dense.landmark_vertices``).

    Raises:
        ValueError: when no frame shows a face.
    """
    if np.isnan(face_landmarks).all(axis=(1, 2)).all():
        raise ValueError("no frame shows a face")
    fitting, faces = config.fitting, config.faces
    device = fitting.device
    model = model.to(device)
    identity_weights = torch.from_numpy(identity).float().to(device)[None]
    centre = model.neutral[ict.SKIN].mean(dim=0)
    landmark_vertices = model.landmarks.to(device)
    mesh = torch.from_numpy(mesh_vertices).long().to(device)
    pupils = model.pupils().to(device)

    width, height = image_size
    scale = fitting.resolution / max(width, height)
    camera = Camera.fixed(width, height, fitting.fov_y_degrees).scaled(scale).to(device)
    targets = landmark_targets(face_landmarks, scale, model, device)
    mesh_targets = targets.mediapipe[:, : dense.NUM_MESH_LANDMARKS, :2].nan_to_num()
    mesh_weights = torch.ones(dense.NUM_MESH_LANDMARKS, device=device)
    mesh_weights[list(dense.FACE_OVAL)] = JAWLINE_WEIGHT
    mesh_weights = mesh_weights * targets.found[:, None]
    params = initial_parameters(
        face_landmarks, targets, camera, model, centre, fitting.eye_distance
    )
    prior = torch.from_numpy(estimate).float().to(device).clamp(0, 1)
    with torch.no_grad():
        params.expression_logits.copy_(torch.logit(prior.clamp(0.02, 0.98)))

    def pose() -> torch.Tensor:
        return head_vertices(
            model, identity_weights, params.expressions(), params.eye_rotations(),
            params.rotations, params.translations, centre,
        )  # fmt: skip

    def loss(refining: bool) -> dict[str, torch.Tensor]:
        vertices = pose()
        terms = {
            "landmarks": fitting.keypoint_weight
            * keypoint_loss(
                camera.project(vertices[:, mesh]), mesh_targets, mesh_weights, camera.height
            ),
            "smoothness": fitting.smoothness_weight
            * (
                second_difference(params.rotations).square().sum(-1).mean()
                + second_difference(params.translations).square().sum(-1).mean()
                + 0.01 * params.expressions().diff(dim=0).square().sum(-1).mean()
            ),
        }
        if refining:
            terms["prior"] = (
                faces.prior_weight * (params.expressions() - prior).square().sum(-1).mean()
            )
            terms["irises"] = fitting.keypoint_weight * keypoint_loss(
                camera.project(vertices[:, pupils]),
                targets.irises,
                targets.found[:, None] * facing(vertices, pupils).detach(),
                camera.height,
            )
            terms["gaze"] = (
                fitting.gaze_prior_weight * params.eye_angles.square().sum((1, 2)).mean()
                + fitting.smoothness_weight
                * second_difference(params.eye_angles).square().sum((1, 2)).mean()
            )
        return terms

    def optimise(trainable: list[torch.Tensor], steps: int, refining: bool) -> None:
        optimizer = torch.optim.Adam(trainable, lr=faces.learning_rate)
        for _ in range(steps):
            optimizer.zero_grad()
            sum(loss(refining).values()).backward()
            optimizer.step()

    rigid = [params.rotations, params.translations]
    pose_steps, refine_steps = faces.iterations
    optimise(rigid, pose_steps, refining=False)
    with torch.no_grad():
        estimate_error = _landmark_error(pose(), camera, landmark_vertices, targets, scale)
    optimise(rigid + [params.expression_logits, params.eye_angles], refine_steps, refining=True)

    with torch.no_grad():
        vertices = pose()
        error = _landmark_error(vertices, camera, landmark_vertices, targets, scale)
        projected = camera.project(vertices[:, mesh])
        distance = (projected - mesh_targets).norm(dim=-1).mean(-1) / scale
        dense_error = torch.where(targets.found > 0, distance, torch.nan)
        outline = _head_outline(vertices, camera, landmark_vertices) / scale
        weights = params.expressions().cpu().numpy()
        return FaceFit(
            frame_indices=frame_indices,
            fps=fps,
            expression_names=np.array(expression_names),
            expressions=match_expressions(weights, model.expression_names, expression_names),
            rotations=params.rotations.cpu().numpy(),
            eye_rotations=_world_eye_rotations(params, model).cpu().numpy(),
            found=(targets.found > 0).cpu().numpy(),
            estimate=match_expressions(estimate, model.expression_names, expression_names),
            landmark_error=error.cpu().numpy(),
            estimate_error=estimate_error.cpu().numpy(),
            dense_error=dense_error.cpu().numpy(),
            outline=outline.cpu().numpy(),
        )


def _head_outline(
    vertices: torch.Tensor, camera: Camera, landmark_vertices: torch.Tensor
) -> torch.Tensor:
    """``(N, 4)`` x0, y0, x1, y1 in pixels around the head from the crown to just below the
    chin: the neck runs on out of the frame."""
    skin = camera.project(vertices[:, : ict.SKIN.stop])
    top = skin[..., 1].amin(dim=1)
    chin = camera.project(vertices[:, landmark_vertices[CHIN]])[:, 1]
    below = chin + 0.1 * (chin - top)
    kept = skin[..., 1] <= below[:, None]
    lower = torch.where(kept[..., None], skin, torch.full_like(skin, torch.inf)).amin(dim=1)
    upper = torch.where(kept[..., None], skin, torch.full_like(skin, -torch.inf)).amax(dim=1)
    return torch.cat((lower, upper), dim=-1)


def _landmark_error(
    vertices: torch.Tensor,
    camera: Camera,
    landmark_vertices: torch.Tensor,
    targets: LandmarkTargets,
    scale: float,
) -> torch.Tensor:
    """``(N,)`` mean landmark distance in pixels of the video, NaN where no face was found."""
    projected = camera.project(vertices[:, landmark_vertices])
    distance = (projected - targets.landmarks).norm(dim=-1).mean(-1) / scale
    return torch.where(targets.found > 0, distance, torch.nan)


def _world_eye_rotations(params: VideoParameters, model: IctFaceKit) -> torch.Tensor:
    """``(N, 2, 3)`` the eyeballs' rotation vectors in world axes, the one on the +x side first.

    The fit's eyeballs come in ``ict.EYEBALL_ORDER`` and turn about the head model's axes."""
    rotations = params.eye_rotations() @ ICT_TO_WORLD.to(params.eye_angles).T
    plus_x_first = (
        model.neutral[ict.LEFT_EYEBALL, 0].mean() > model.neutral[ict.RIGHT_EYEBALL, 0].mean()
    )
    return rotations if plus_x_first else rotations.flip(1)
