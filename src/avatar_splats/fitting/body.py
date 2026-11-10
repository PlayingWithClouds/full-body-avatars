"""Fitting Anny to the body turntable: one shape, one pose with small per-frame movements of the
head, spine and limbs, and a turntable angle per frame.

The shape is Anny's phenotype plus MakeHuman's local changes of the limbs and torso (breast
volume, upper arm fat, hip width, ...; see ``anny.BODY_CHANGES``), all smooth by construction.
Per-vertex offsets fitted to silhouettes did worse: wherever the pose was a little off they warped
the surface to match the outline instead, a step at the elbow, boxy breasts.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import roma
import torch
import torch.nn.functional as F

from avatar_splats.body_models.anny import AnnyBody
from avatar_splats.body_models.measurements import foot_ends
from avatar_splats.body_models.skinning import linear_blend_skinning, sparsify_weights
from avatar_splats.cameras import Camera, body_yaw
from avatar_splats.config import FittingConfig
from avatar_splats.fitting.losses import keypoint_loss, second_difference, silhouette_loss
from avatar_splats.fitting.silhouette import SurfaceSamples, render_silhouette, silhouette_iou
from avatar_splats.preprocess import Preprocessed
from avatar_splats.records import ArrayRecord
from avatar_splats.rendering import rasterize

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
        **{f"finger{finger}-{joint}": "LR" for finger in range(1, 6) for joint in range(1, 4)},
    }.items()
    for side in (sides or [""])
)  # fmt: skip
"""Bones whose rotation is fitted; toes, metacarpals and the rest keep Anny's rest pose."""

MOVING_BONES = (
    "spine05", "spine04", "spine03", "spine02", "spine01", "neck01", "neck02", "neck03", "head",
    "clavicle.L", "clavicle.R", "upperarm01.L", "upperarm01.R", "lowerarm01.L", "lowerarm01.R",
    "upperleg01.L", "upperleg01.R",
)  # fmt: skip
"""Bones that may move from frame to frame, smoothly. The videos are generated, not a rigid
turntable: the performer sways and her arms drift by a few centimetres as she turns, and with one
pose for all frames the Gaussians blur over the difference."""

MIN_KEYPOINT_CONFIDENCE = 0.3
MIN_HAND_CONFIDENCE = 0.5
BONE_INFLUENCES = 8
"""Bones per vertex when skinning Anny during fitting; more change nothing visible."""


@dataclass(frozen=True)
class BodyFit(ArrayRecord):
    """Anny's parameters for every frame of the body video."""

    frame_indices: np.ndarray
    """``(N,)`` the video frames."""
    phenotype: np.ndarray
    """``(P,)`` Anny phenotype values in [0, 1]."""
    local_changes: np.ndarray
    """``(L,)`` Anny local changes in [-1, 1], one per ``AnnyBody.local_change_labels``."""
    rotations: np.ndarray
    """``(N, J, 3)`` rotation vectors per bone and frame, root first."""
    translation: np.ndarray
    """``(N, 3)`` root translation per frame, in metres."""
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


class HandJoints:
    """Anny's points matching MediaPipe's 21 hand landmarks, on both hands.

    The wrist and the finger joints are bone heads; Anny's finger bones run from the thumb
    (``finger1``) to the little finger (``finger5``), three per finger from the knuckle nearest
    the wrist, as the landmarks do. Fingertips have no bone, so each is the vertex of the last
    finger bone farthest along it.
    """

    def __init__(self, body: AnnyBody, rest: torch.Tensor, rest_bone_poses: torch.Tensor):
        """Args: ``(V, 3)`` rest vertices and ``(J, 4, 4)`` rest bone frames of the body."""
        heads = rest_bone_poses[:, :3, 3]
        self.bones, self.tips = [], []
        for side in "LR":
            bones = [body.bone(f"wrist.{side}")]
            for finger in range(1, 6):
                joints = [body.bone(f"finger{finger}-{joint}.{side}") for joint in (1, 2, 3)]
                bones += joints
                direction = heads[joints[2]] - heads[joints[1]]
                along = (rest - heads[joints[2]]) @ direction
                along[body.skin_weights[:, joints[2]] < 0.5] = -torch.inf
                self.tips.append(int(along.argmax()))
            self.bones.append(bones)

    def __call__(self, vertices: torch.Tensor, joints: torch.Tensor) -> torch.Tensor:
        """``(N, 2, 21, 3)`` hand points of ``(N, V, 3)`` posed vertices and ``(N, J, 3)`` posed
        bone heads, the left hand first."""
        hands = []
        for side, bones in enumerate(self.bones):
            points = [joints[:, bones[0]]]
            for finger in range(5):
                points += [joints[:, bone] for bone in bones[1 + 3 * finger : 4 + 3 * finger]]
                points.append(vertices[:, self.tips[5 * side + finger]])
            hands.append(torch.stack(points, dim=1))
        return torch.stack(hands, dim=1)


def badly_fitted(iou: np.ndarray, spread: float, most: float) -> np.ndarray:
    """``(N,)`` bool: the frames whose ``(N,)`` silhouette IoU falls more than ``spread`` robust
    standard deviations (1.4826 times the median absolute deviation) below the median, at most
    the worst ``most`` share of them; frames without an IoU (NaN) are among them."""
    known = np.isfinite(iou)
    median = np.median(iou[known])
    deviation = 1.4826 * np.median(np.abs(iou[known] - median))
    low = ~known | (iou < median - spread * deviation)
    worst = np.argsort(np.where(known, iou, -np.inf))[: int(most * len(iou))]
    out = np.zeros(len(iou), bool)
    out[worst] = low[worst]
    return out


RELAXED_CURL = 10.0
"""Degrees each finger joint of a relaxed hand bends towards the palm."""


@torch.no_grad()
def relaxed_hands(body: AnnyBody, rest_bone_poses: torch.Tensor) -> torch.Tensor:
    """``(J, 3)`` Anny ``local-ref`` rotations of an open, relaxed hand (zero elsewhere): the
    fingers brought together alongside the middle finger (Anny's zero pose spreads them) and
    each joint bent ``RELAXED_CURL`` towards the palm; the thumb kept.

    Each finger's sideways turn is searched for (its knuckle's z, whose direction differs by
    side), the curl is +x on both hands.
    """
    device = rest_bone_poses.device
    axes = rest_bone_poses[:, :3, :3]
    zeros = torch.zeros(1, body.num_bones, 3, device=device)
    angles = torch.deg2rad(torch.linspace(-30, 30, 121, device=device))
    out = torch.zeros(body.num_bones, 3, device=device)

    def direction(rotations: torch.Tensor, finger: int, side: str) -> torch.Tensor:
        # (B, 3) a finger's direction, knuckle to last joint, for (B, J, 3) rotations
        skinning = body.skinning(rest_bone_poses, rotations, zeros[:, 0].expand(len(rotations), 3))
        heads = (skinning @ rest_bone_poses)[..., :3, 3]
        first, last = (body.bone(f"finger{finger}-{j}.{side}") for j in (1, 3))
        return F.normalize(heads[:, last] - heads[:, first], dim=-1)

    for side in "LR":
        middle = direction(zeros, 3, side)[0]
        for finger in (2, 4, 5):
            knuckle = body.bone(f"finger{finger}-1.{side}")
            tried = zeros.repeat(len(angles), 1, 1)
            tried[:, knuckle] = angles[:, None] * axes[knuckle, :, 2]
            best = int((direction(tried, finger, side) @ middle).argmax())
            out[knuckle] = angles[best] * axes[knuckle, :, 2]
        for finger in range(2, 6):
            for joint in (1, 2, 3):
                bone = body.bone(f"finger{finger}-{joint}.{side}")
                out[bone] += math.radians(RELAXED_CURL) * axes[bone, :, 0]
    return out


def surface_keypoint_vertices(
    body: AnnyBody, rest: torch.Tensor, rest_bone_poses: torch.Tensor
) -> torch.Tensor:
    """``(12,)`` the vertices of Anny's matching ``preprocess.keypoints.EXTRA_KEYPOINTS``, picked
    on ``(V, 3)`` rest vertices with ``(J, 4, 4)`` rest bone frames (the body faces -y, z up):

    - the big and small toe: the foremost vertex of the first and the fifth toe,
    - the heel: the hindmost of the foot's lowest third,
    - the olecranon and the cubital fossa: the vertices near the elbow farthest behind and in
      front of it, across the bend (a straight arm bends forwards),
    - the acromion: the highest vertex near the shoulder joint.
    """
    heads = rest_bone_poses[:, :3, 3]
    dominant = body.skin_weights.argmax(dim=1)

    def of(*bones: str) -> torch.Tensor:
        return torch.isin(dominant, torch.tensor([body.bone(b) for b in bones], device=rest.device))

    def best(score: torch.Tensor, where: torch.Tensor) -> int:
        return int(torch.where(where, score, -torch.inf).argmax())

    feet = [vertex for ends in foot_ends(body, rest) for vertex in ends]
    olecranon, fossa, acromion = [], [], []
    for side in "LR":
        shoulder, elbow, wrist = (
            heads[body.bone(f"{b}.{side}")] for b in ("upperarm01", "lowerarm01", "wrist")
        )
        down = F.normalize(elbow - shoulder, dim=0)
        inside = (wrist - elbow) / (wrist - elbow).norm() - down + rest.new_tensor([0, -0.5, 0])
        inside = F.normalize(inside - (inside @ down) * down, dim=0)
        arm = of(*(f"{b}.{side}" for b in ("upperarm01", "upperarm02", "lowerarm01", "lowerarm02")))
        near_elbow = arm & ((rest - elbow).norm(dim=1) < 0.06)
        olecranon.append(best(-(rest - elbow) @ inside, near_elbow))
        fossa.append(best((rest - elbow) @ inside, near_elbow))
        near_shoulder = (rest - shoulder).norm(dim=1) < 0.08
        top = of(*(f"{b}.{side}" for b in ("clavicle", "shoulder01", "upperarm01")))
        acromion.append(best(rest[:, 2], near_shoulder & top))
    return torch.tensor(feet + olecranon + fossa + acromion, device=rest.device)


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
        self.local_change_raw = torch.nn.Parameter(torch.zeros(len(body.local_change_labels)))
        self.register_buffer("posed", torch.zeros(bones, 1))
        self.posed[[body.bone(name) for name in POSED_BONES]] = 1
        self.moving = [body.bone(name) for name in MOVING_BONES]

    def phenotype(self) -> torch.Tensor:
        return torch.sigmoid(self.phenotype_logit)

    def local_changes(self) -> torch.Tensor:
        return torch.tanh(self.local_change_raw)

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
    local_changes: torch.Tensor,
    rotations: torch.Tensor,
    translation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Anny posed, skinned with ``BONE_INFLUENCES`` bones per vertex.

    Args:
        phenotype: ``(1, P)`` shared by all frames.
        local_changes: ``(1, L)`` shared by all frames.
        rotations: ``(N, J, 3)`` rotation vectors.
        translation: ``(N, 3)`` root translations.

    Returns:
        ``(N, J, 4, 4)`` skinning matrices and ``(N, V, 3)`` vertices.
    """
    out = body(phenotype, local_changes, rotations, translation)
    indices, weights = sparsify_weights(body.skin_weights, BONE_INFLUENCES)
    vertices = linear_blend_skinning(out["rest_vertices"], indices, weights, out["skinning"])
    return out["skinning"], vertices


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
        tensor(fit.local_changes)[None],
        tensor(fit.rotations[frames]),
        tensor(fit.translation[frames]),
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

    def rest_pose() -> dict[str, torch.Tensor]:
        bones = torch.zeros(1, body.num_bones, 3, device=device)
        return body(params.phenotype()[None], params.local_changes()[None], bones, bones[:, 0])

    rest = rest_pose()
    samples = SurfaceSamples.draw(
        rest["rest_vertices"][0].detach(), body.faces, config.surface_samples
    )
    radius = samples.radius
    inset = config.silhouette_inset * radius

    hand_joints = HandJoints(body, rest["rest_vertices"][0], rest["rest_bone_poses"][0])
    hand_targets = torch.from_numpy(np.nan_to_num(preprocessed.hands)).float().to(device)
    hand_confidence = hand_targets[..., 2] * (hand_targets[..., 2] >= MIN_HAND_CONFIDENCE)
    hand_targets = hand_targets[..., :2] * scale

    def pose() -> tuple[torch.Tensor, torch.Tensor]:
        """``(N, V, 3)`` posed vertices and ``(N, 2, 21, 3)`` hand points."""
        rest = rest_pose()
        skinning, vertices = pose_body(
            body, params.phenotype()[None], params.local_changes()[None], params.rotations(),
            params.translations(),
        )  # fmt: skip
        heads = rest["rest_bone_poses"][0, :, :, 3]
        joints = (skinning @ heads.unsqueeze(-1))[..., :3, 0]
        return vertices, hand_joints(vertices, joints)

    def projected_keypoints(vertices: torch.Tensor) -> torch.Tensor:
        return camera.project(body.keypoint_regressor({"vertices": vertices}))

    def height() -> torch.Tensor:
        """Height of the rest pose: a tall body far away and a short one close by look the same,
        so the configured height sets the scale."""
        z = rest_pose()["rest_vertices"][0, :, 2]
        return z.max() - z.min()

    def loss(phase: int) -> dict[str, torch.Tensor]:
        vertices, hands = pose()
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
            "shape_prior": config.local_change_weight * params.local_changes().square().sum(),
        }
        if phase >= 1:
            terms["hands"] = config.hand_weight * keypoint_loss(
                camera.project(hands), hand_targets, hand_confidence, camera.height
            )
            points = samples.points(vertices, body.faces, inset)
            silhouettes = render_silhouette(points, radius, camera)
            terms["silhouette"] = config.silhouette_weight * silhouette_loss(silhouettes, masks)
        return terms

    placement = [params.yaw, params.tilt, params.translation, params.translation_residual]
    placement.append(params.phenotype_logit)
    shape = [params.local_change_raw]
    phases = (
        [{"params": placement}],
        [{"params": placement + [params.pose, params.movement]}],
        [{"params": placement + shape + [params.pose, params.movement]}],
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
        vertices, _ = pose()
        # The mesh's own outline, not the blobs': it is what the avatar shows.
        silhouettes = torch.stack([
            rasterize(camera.screen(v), body.faces, camera.width, camera.height).mask().float()
            for v in vertices
        ])  # fmt: skip
        distance = (projected_keypoints(vertices) - targets).norm(dim=-1) / scale
        seen = confidence > 0
        error = (distance * seen).sum(-1) / seen.sum(-1).clamp_min(1)
        return BodyFit(
            frame_indices=preprocessed.frame_indices,
            phenotype=params.phenotype().cpu().numpy(),
            local_changes=params.local_changes().cpu().numpy(),
            rotations=params.rotations().cpu().numpy(),
            translation=params.translations().cpu().numpy(),
            fov_y_degrees=config.fov_y_degrees,
            image_size=np.array(image_size),
            silhouette_iou=silhouette_iou(silhouettes, masks).cpu().numpy(),
            keypoint_error=error.cpu().numpy(),
        )
