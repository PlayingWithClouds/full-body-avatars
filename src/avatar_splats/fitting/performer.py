"""What ``PerformerConfig`` holds, estimated from a few frames of a video where it is not known:
her height, measurements, shoe size and the breasts' cupsize and firmness.

Anny is fitted to frames that show her from the front (``frontal_scores``), one shape for all
of them and a pose each, at a nominal height: one camera cannot tell a tall person far away
from a short one close by. The scale comes afterwards from things whose real size varies little
between adults, each a vote (``scale_votes``): the iris's diameter, the distance between the
pupils (MediaPipe's landmarks, against the fitted head's depth), and Depth Pro's metric depth of
her (against the fitted body's). The measurements are those of the fitted mesh, scaled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import nvdiffrast.torch as dr
import torch
import torch.nn.functional as F

from avatar_splats.body_models.anny import AnnyBody
from avatar_splats.body_models.skinning import linear_blend_skinning, sparsify_weights
from avatar_splats.cameras import Camera
from avatar_splats.fitting.body import BONE_INFLUENCES, POSED_BONES, surface_keypoint_vertices
from avatar_splats.fitting.losses import depth_loss, keypoint_loss, normal_loss, silhouette_loss
from avatar_splats.fitting.silhouette import SurfaceSamples, render_silhouette
from avatar_splats.motion.priors import JointLimits
from avatar_splats.rendering.surface import render_surface

IRIS_DIAMETER = 0.0117
"""Metres: the iris's visible diameter, about the same in every adult (sd ~0.5 mm)."""
PUPIL_DISTANCE = 0.062
"""Metres: the distance between the pupils, women's mean (sd ~3.5 mm)."""
IRISES = ((468, 469, 470, 471, 472), (473, 474, 475, 476, 477))
"""MediaPipe's iris landmarks per eye: the centre, then four on the rim, round it."""
FRONTAL_WIDTH = (0.7, 0.55)
"""Shoulder and hip width over the torso's length (shoulders' to hips' middle) when facing the
camera: less means she is turned away."""


def frontal_scores(keypoints: np.ndarray, min_confidence: float) -> np.ndarray:
    """``(N,)`` in [0, 1]: how squarely ``(N, 17, 3)`` COCO keypoints (x, y in pixels,
    confidence) face the camera, upright and whole: every keypoint placed, her left (COCO's
    left) on the image's right, shoulders and hips as wide as from the front, the torso upright.
    0 where any keypoint is missing."""
    xy = np.nan_to_num(keypoints[..., :2])
    placed = (np.nan_to_num(keypoints[..., 2]) >= min_confidence).all(-1)
    shoulders, hips = xy[:, [5, 6]], xy[:, [11, 12]]
    torso = hips.mean(1) - shoulders.mean(1)  # (N, 2) shoulders' middle to hips'
    length = np.linalg.norm(torso, axis=-1).clip(1e-6)
    widths = [
        np.clip((pair[:, 0, 0] - pair[:, 1, 0]) / length / share, 0, 1)
        for pair, share in zip((shoulders, hips), FRONTAL_WIDTH, strict=True)
    ]
    upright = np.clip(torso[:, 1] / length, 0, 1)  # down the image
    return np.where(placed, widths[0] * widths[1] * upright, 0.0)


def iris_diameters(face_landmarks: np.ndarray) -> np.ndarray:
    """``(N,)`` the irises' diameter in pixels, both eyes' mean, from ``(N, 478, 3)`` MediaPipe
    landmarks: the longer of each iris's two rim diameters (turning the head shortens one)."""
    diameters = []
    for _, *rim in IRISES:
        points = face_landmarks[:, rim, :2]
        across = np.linalg.norm(points[:, 0] - points[:, 2], axis=-1)
        down = np.linalg.norm(points[:, 1] - points[:, 3], axis=-1)
        diameters.append(np.maximum(across, down))
    return np.mean(diameters, axis=0)


def pupil_distances(face_landmarks: np.ndarray) -> np.ndarray:
    """``(N,)`` the distance between the irises' centres in pixels, in 3D (MediaPipe's depth is
    in the same pixels), so a turned head does not shorten it."""
    left, right = (face_landmarks[:, eye[0]] for eye in IRISES)
    return np.linalg.norm(left - right, axis=-1)


def scale_from_size(pixels: np.ndarray, depth: np.ndarray, focal: float, real: float) -> np.ndarray:
    """How many times larger the world is than the fit: ``real`` metres against what something
    ``pixels`` long at ``depth`` metres (the fit's) measures through a lens of ``focal``
    pixels."""
    return real * focal / (np.asarray(pixels) * np.asarray(depth))


def eu_shoe_size(foot: float) -> float:
    """The EU shoe size of a foot ``foot`` metres long, heel to big toe (the config's rule:
    the foot in centimetres is the size / 1.5 - 1.5)."""
    return 1.5 * (100 * foot + 1.5)


def foot_length(eu_size: float) -> float:
    """Metres heel to big toe of an EU shoe size (``eu_shoe_size``'s inverse)."""
    return (eu_size / 1.5 - 1.5) / 100


def scaled_measurements(measured: dict[str, float], scale: float) -> dict[str, float]:
    """Measurements of a body ``scale`` times as large: lengths times it, weight its cube."""
    return {k: v * scale**3 if k == "weight" else v * scale for k, v in measured.items()}


@dataclass(frozen=True)
class FrameTargets:
    """What the frames show, at the fit camera's image size ``(H, W)``."""

    masks: torch.Tensor
    """``(N, H, W)`` her outline, in [0, 1]."""
    compared: torch.Tensor
    """``(N, H, W)`` 1 where outlines are compared (not hair or clothes)."""
    keypoints: torch.Tensor
    """``(N, 17, 3)`` COCO keypoints, x, y in the camera's pixels and confidence (0 where they
    do not count)."""
    extra_keypoints: torch.Tensor
    """``(N, 12, 3)`` the ``EXTRA_KEYPOINTS`` likewise."""
    normals: torch.Tensor
    """``(N, H, W, 3)`` estimated unit normals in the camera's axes; zero where unknown."""
    depths: torch.Tensor
    """``(N, D, H, W)`` D estimates of depth, metres up to scale; NaN where unknown."""
    surface_region: torch.Tensor
    """``(N, H, W)`` where normals and depth are compared: inside the outline, off its edge."""


@dataclass(frozen=True)
class ShapeFitSettings:
    iterations: int
    shape_learning_rate: float
    pose_learning_rate: float
    weights: dict[str, float]
    """Per term of ``fit_frames_shape``: keypoints, silhouette, normals, depth, height,
    phenotype_prior (per squared phenotype step from the start), local_changes, pose_prior (per
    radian² and decimetre² of the pose corrections), joint_limits."""
    held: tuple[str, ...]
    """Phenotype labels kept as they start."""
    surface_samples: int
    silhouette_inset: float
    """As ``FittingConfig.silhouette_inset``."""


@dataclass(frozen=True)
class FramesShape:
    """One shape posed in N frames."""

    phenotype: np.ndarray
    """``(P,)``"""
    local_changes: np.ndarray
    """``(L,)``"""
    rotations: np.ndarray
    """``(N, J, 3)``"""
    translation: np.ndarray
    """``(N, 3)`` metres."""
    silhouette_iou: np.ndarray | None = None
    """``(N,)`` outside the ignored pixels."""
    keypoint_error: np.ndarray | None = None
    """``(N,)`` mean distance in the camera's pixels."""


def fit_frames_shape(
    body: AnnyBody,
    start: FramesShape,
    targets: FrameTargets,
    camera: Camera,
    height: float,
    settings: ShapeFitSettings,
    near: float,
    far: float,
    log_every: int = 100,
) -> FramesShape:
    """Anny's phenotype and local changes for the frames, ``height`` metres tall, the poses
    (from a motion fit) corrected a little as it changes. Against the keypoints, the outline
    away from hair and clothes, and the estimated normals and depth, which see the breasts'
    shape that the outline from the front does not."""
    device = body.faces.device
    s, w = settings, settings.weights
    rotations = torch.from_numpy(start.rotations).float().to(device)
    translation = torch.from_numpy(start.translation).float().to(device)
    start_phenotype = torch.from_numpy(start.phenotype).float().to(device)
    phenotype = torch.nn.Parameter(torch.logit(start_phenotype.clamp(1e-3, 1 - 1e-3)))
    changes = torch.nn.Parameter(
        torch.atanh(torch.from_numpy(start.local_changes).float().to(device).clamp(-0.999, 0.999))
    )
    free = torch.tensor([label not in s.held for label in body.phenotype_labels], device=device)
    phenotype.register_hook(lambda grad: grad * free)
    corrected = torch.zeros(body.num_bones, 1, device=device)
    corrected[[body.bone(n) for n in POSED_BONES if not n.startswith("finger")] + [0]] = 1
    turn = torch.nn.Parameter(torch.zeros_like(rotations))
    shift = torch.nn.Parameter(torch.zeros_like(translation))
    bone_indices, bone_weights = sparsify_weights(body.skin_weights, BONE_INFLUENCES)
    context = dr.RasterizeCudaContext()

    def posed():
        out = body(
            torch.sigmoid(phenotype)[None],
            torch.tanh(changes)[None],
            rotations + turn * corrected,
            translation + shift,
        )
        vertices = linear_blend_skinning(
            out["rest_vertices"], bone_indices, bone_weights, out["skinning"]
        )
        return out, vertices

    with torch.no_grad():
        out, _ = posed()
        rest = out["rest_vertices"][0]
        samples = SurfaceSamples.draw(rest, body.faces, s.surface_samples)
        limits = JointLimits(body, out["rest_bone_poses"][0])
        extra_vertices = surface_keypoint_vertices(body, rest, out["rest_bone_poses"][0])
    inset = s.silhouette_inset * samples.radius
    points, extra = targets.keypoints, targets.extra_keypoints

    def terms():
        out, vertices = posed()
        z = out["rest_vertices"][0, :, 2]
        projected = camera.project(body.keypoint_regressor({"vertices": vertices}))
        silhouettes = render_silhouette(
            samples.points(vertices, body.faces, inset), samples.radius, camera
        )
        drawn = render_surface(context, vertices, body.faces, camera, near, far)
        region = targets.surface_region * drawn.covered
        depths = targets.depths.unbind(1)
        return {
            "keypoints": w["keypoints"]
            * (
                keypoint_loss(projected, points[..., :2], points[..., 2], camera.height)
                + keypoint_loss(
                    camera.project(vertices[:, extra_vertices]),
                    extra[..., :2],
                    extra[..., 2],
                    camera.height,
                )
            ),
            "silhouette": w["silhouette"]
            * silhouette_loss(silhouettes, targets.masks, targets.compared),
            "normals": w["normals"] * normal_loss(drawn.normals, targets.normals, region),
            "depth": w["depth"]
            * sum(depth_loss(drawn.depth, d, region) for d in depths)
            / len(depths),
            "height": w["height"] * ((z.max() - z.min()) / height - 1).square(),
            "phenotype_prior": w["phenotype_prior"]
            * ((torch.sigmoid(phenotype) - start_phenotype) * free).square().sum(),
            "local_changes": w["local_changes"] * torch.tanh(changes).square().sum(),
            "pose_prior": w["pose_prior"]
            * (turn.square().sum((1, 2)).mean() + (10 * shift).square().sum(-1).mean()),
            "joint_limits": w["joint_limits"] * limits(rotations + turn * corrected).mean(),
        }

    optimizer = torch.optim.Adam(
        [
            {"params": [phenotype, changes], "lr": s.shape_learning_rate},
            {"params": [turn, shift], "lr": s.pose_learning_rate},
        ]
    )
    for iteration in range(s.iterations):
        values = terms()
        optimizer.zero_grad()
        sum(values.values()).backward()
        optimizer.step()
        if iteration % log_every == 0 or iteration == s.iterations - 1:
            print(f"{iteration:4d}: " + ", ".join(f"{k} {v.item():.2e}" for k, v in values.items()))

    with torch.no_grad():
        _, vertices = posed()
        silhouettes = render_silhouette(
            samples.points(vertices, body.faces, inset), samples.radius, camera
        )
        seen = targets.compared > 0.5
        drawn_, mask = (silhouettes > 0.5) & seen, (targets.masks > 0.5) & seen
        iou = (drawn_ & mask).sum((1, 2)) / (drawn_ | mask).sum((1, 2)).clamp_min(1)
        projected = camera.project(body.keypoint_regressor({"vertices": vertices}))
        sure = points[..., 2] > 0
        distance = (projected - points[..., :2]).norm(dim=-1) * sure
        error = distance.sum(-1) / sure.sum(-1).clamp_min(1)
        return FramesShape(
            phenotype=torch.sigmoid(phenotype).cpu().numpy(),
            local_changes=torch.tanh(changes).cpu().numpy(),
            rotations=(rotations + turn * corrected).cpu().numpy(),
            translation=(translation + shift).cpu().numpy(),
            silhouette_iou=iou.cpu().numpy(),
            keypoint_error=error.cpu().numpy(),
        )


def scale_votes(
    body: AnnyBody,
    shape: FramesShape,
    camera: Camera,
    face_landmarks: np.ndarray,
    depth_pro: torch.Tensor,
    person: torch.Tensor,
    near: float,
    far: float,
) -> dict[str, np.ndarray]:
    """Per anchor, ``(N,)`` per frame how many times larger she is than the fit (NaN where the
    frame cannot tell).

    Args:
        camera: the frames' own camera (pixels of ``face_landmarks``).
        face_landmarks: ``(N, 478, 3)`` MediaPipe's, x, y and depth in pixels; NaN where none.
        depth_pro: ``(N, h, w)`` Depth Pro's metric depth, at any size.
        person: ``(N, h, w)`` bool, her pixels at that size, away from hair and clothes.
    """
    device = body.faces.device

    def tensor(array):
        return torch.from_numpy(np.asarray(array)).float().to(device)

    with torch.no_grad():
        out = body(
            tensor(shape.phenotype)[None],
            tensor(shape.local_changes)[None],
            tensor(shape.rotations),
            tensor(shape.translation),
        )
        indices, weights = sparsify_weights(body.skin_weights, BONE_INFLUENCES)
        vertices = linear_blend_skinning(out["rest_vertices"], indices, weights, out["skinning"])
        eyes = camera.screen(body.keypoint_regressor({"vertices": vertices})[:, 1:3])
        eye_depth = eyes[..., 2].mean(-1).cpu().numpy()  # (N,) metres, the fit's
        K = camera.K.clone()
        K[:2] *= person.shape[-1] / camera.width
        map_camera = Camera(K, camera.view, person.shape[-1], person.shape[-2])
        drawn = render_surface(
            dr.RasterizeCudaContext(), vertices, body.faces, map_camera, near, far
        )
        inside = person & (drawn.covered > 0) & torch.isfinite(depth_pro)
        ratios = torch.where(inside, depth_pro / drawn.depth.clamp_min(1e-6), torch.nan)
        depth_scale = ratios.flatten(1).nanmedian(dim=1).values.cpu().numpy()
    focal = float(camera.K[1, 1])
    return {
        "iris": scale_from_size(iris_diameters(face_landmarks), eye_depth, focal, IRIS_DIAMETER),
        "pupils": scale_from_size(
            pupil_distances(face_landmarks), eye_depth, focal, PUPIL_DISTANCE
        ),
        "Depth Pro": depth_scale,
    }


def anny_height(body: AnnyBody, phenotype: np.ndarray, local_changes: np.ndarray) -> float:
    """Metres from the soles to the top of the head of Anny's rest mesh."""
    device = body.faces.device
    zeros = torch.zeros(1, body.num_bones, 3, device=device)
    with torch.no_grad():
        rest = body(
            torch.from_numpy(phenotype).float().to(device)[None],
            torch.from_numpy(local_changes).float().to(device)[None],
            zeros,
            zeros[:, 0],
        )["rest_vertices"][0]
    return float(rest[:, 2].max() - rest[:, 2].min())


def resized(maps: torch.Tensor, size: tuple[int, int], mode: str = "bilinear") -> torch.Tensor:
    """``(N, h, w)`` or ``(N, h, w, C)`` maps at ``size`` ``(H, W)``."""
    channels = maps.dim() == 4
    x = maps.permute(0, 3, 1, 2) if channels else maps[:, None]
    x = F.interpolate(x.float(), size, mode=mode)
    return x.permute(0, 2, 3, 1) if channels else x[:, 0]
