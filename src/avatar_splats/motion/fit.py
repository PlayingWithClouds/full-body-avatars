"""Fitting the avatar's body to every person in a motion video, frame by frame.

Each person is the performer's own Anny body (the fitted phenotype and local changes), so the
motion lands on the avatar's skeleton as it is, with no retargeting. Per frame and person the fit
finds the root's rotation and translation and every posed bone's rotation (the fingers' where
``MotionConfig.pose_fingers``; else they stay open and the hands' points only turn the hands),
matching the detected keypoints and hands, and in the last phase the silhouettes. Where the
observations carry estimated surface normals and depths, the body's are matched to them from the
second phase on: which way a limb turns and which is in front, which keypoints and outlines do
not tell. Where they carry the estimated depth of the skin at the keypoints' pixels, the fit
starts from the 3D points these give (a leg folded towards the camera or away is no longer a
guess), and keeps the keypoints behind that skin and near it throughout.

One camera sees no depth, which the terms besides the keypoints make up for as far as they can:
smooth motion, a small pose prior, joint limits, knees that bend only backwards, and bodies that
do not pass through themselves or each other. A person's silhouette is compared only
where no one else's mask is, since whoever is in front hides them there.

Long videos are fitted in overlapping windows of frames, each started from where the one before
left off and blended into it across the overlap, so memory and time grow with the length and the
motion stays continuous. Keypoints, hand points and joints are linear in the bones' skinning
matrices, so they come straight from the bones (``skinned_point_weights``); only the silhouettes
need the skinned mesh, rendered a chunk of frames at a time. Each phase stops when its loss no
longer falls.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import roma
import scipy.ndimage
import torch

from avatar_splats.body_models.anny import AnnyBody
from avatar_splats.body_models.skinning import (
    linear_blend_skinning,
    skinned_point_weights,
    skinned_points,
    sparsify_weights,
)
from avatar_splats.cameras import Camera
from avatar_splats.config import MotionConfig
from avatar_splats.fitting.body import (
    BONE_INFLUENCES,
    MIN_HAND_CONFIDENCE,
    MIN_KEYPOINT_CONFIDENCE,
    POSED_BONES,
    BodyFit,
    HandJoints,
    pose_body,
    relaxed_hands,
    surface_keypoint_vertices,
)
from avatar_splats.fitting.losses import (
    depth_loss,
    keypoint_depth_loss,
    keypoint_loss,
    normal_loss,
    second_difference,
    silhouette_loss,
)
from avatar_splats.fitting.silhouette import SurfaceSamples, render_silhouette, silhouette_iou
from avatar_splats.motion.observe import Observations
from avatar_splats.motion.placement import orientations, place_people
from avatar_splats.motion.priors import (
    BodyProxies,
    JointLimits,
    knee_bend,
    overlap,
    self_overlap,
)
from avatar_splats.records import ArrayRecord
from avatar_splats.rendering.surface import render_surface

logger = logging.getLogger(__name__)

LEG_BONES = ("upperleg01", "lowerleg01", "foot")
"""The bones whose heads are a leg's hip, knee and ankle."""
SURFACE_EDGE = 2
"""Pixels of the mask's edge where estimated normals and depths are not compared: they blur
into the background there."""
NEAR, FAR = 0.05, 50.0
"""Metres: the rasterizer's clipping planes."""


@dataclass(frozen=True)
class MotionFit(ArrayRecord):
    """Anny's pose per person and frame of a motion video, in the video camera's world.

    The camera is the fixed one at the origin looking along +y (``Camera.fixed``).
    """

    frame_indices: np.ndarray
    """``(N,)`` the video frames."""
    fps: float
    fov_y_degrees: float
    image_size: np.ndarray
    """``(2,)`` width and height of the video."""
    phenotype: np.ndarray
    """``(P_,)`` the avatar's Anny phenotype, which every person was fitted with."""
    local_changes: np.ndarray
    """``(L,)`` the avatar's Anny local changes."""
    rotations: np.ndarray
    """``(P, N, J, 3)`` Anny ``local-ref`` rotation vectors per person, frame and bone."""
    translation: np.ndarray
    """``(P, N, 3)`` root translation per person and frame, in metres."""
    expressions: np.ndarray
    """``(P, N, E)`` the avatar's expression weights per person and frame."""
    expression_names: np.ndarray
    """``(E,)`` the avatar's expression names."""
    seen: np.ndarray
    """``(P, N)`` bool: whether the person's keypoints were found in the frame; elsewhere the
    pose is only interpolated."""
    keypoint_error: np.ndarray
    """``(P, N)`` mean distance in pixels from the fitted to the detected keypoints."""
    silhouette_iou: np.ndarray
    """``(P, N)`` IoU of the fitted silhouette with the mask, outside other people's masks and
    the ignored pixels."""

    def camera(self) -> Camera:
        width, height = self.image_size
        return Camera.fixed(int(width), int(height), self.fov_y_degrees)


class _MotionParameters(torch.nn.Module):
    """Rotations and translations of every person in the frames of one window.

    Roots turn from their starting orientation by a rotation of their own, so the parameter
    stays small and smooth even where a body has turned half around.
    """

    def __init__(self, posed: torch.Tensor, rotations: torch.Tensor, translation: torch.Tensor):
        """Start from ``(P, n, J, 3)`` rotation vectors and ``(P, n, 3)`` translations; only
        bones with ``(J, 1)`` ``posed`` set turn, the others keep their start."""
        super().__init__()
        self.register_buffer("posed", posed)
        self.register_buffer("kept", rotations * (1 - posed))
        self.register_buffer("start_rotation", roma.rotvec_to_rotmat(rotations[:, :, 0]))
        self.root = torch.nn.Parameter(torch.zeros_like(rotations[:, :, 0]))
        self.translation = torch.nn.Parameter(translation.clone())
        self.pose = torch.nn.Parameter(rotations * posed)

    def root_rotation(self) -> torch.Tensor:
        """``(P, n, 3, 3)`` the roots' rotations."""
        return roma.rotvec_to_rotmat(self.root) @ self.start_rotation

    def rotations(self, frames: slice = slice(None)) -> torch.Tensor:
        """``(P, n, J, 3)`` rotation vectors of the frames."""
        rotations = self.pose[:, frames] * self.posed + self.kept[:, frames]
        root = roma.rotmat_to_rotvec(self.root_rotation()[:, frames])
        return torch.cat((root.unsqueeze(2), rotations[:, :, 1:]), dim=2)


class _Fitter:
    """What every window of one video is fitted against: the body, the camera and the targets."""

    def __init__(
        self,
        observations: Observations,
        body: AnnyBody,
        shape: BodyFit,
        config: MotionConfig,
        silhouette_inset: float,
        independent: bool = False,
    ):
        self.observations, self.body, self.config = observations, body, config
        self.independent = independent
        device = self.device = config.device
        width, height = (int(v) for v in observations.image_size)
        mask_width = int(observations.mask_size[0])
        self.scale = mask_width / width
        self.camera = (
            Camera.fixed(width, height, config.fov_y_degrees).scaled(self.scale).to(device)
        )
        self.people = observations.num_people

        self.phenotype = self.tensor(shape.phenotype)[None]
        self.local_changes = self.tensor(shape.local_changes)[None]
        zeros = torch.zeros(1, body.num_bones, 3, device=device)
        rest = body(self.phenotype, self.local_changes, zeros, zeros[:, 0])
        self.rest_vertices = rest["rest_vertices"][0]
        self.rest_bones = rest["rest_bone_poses"][0]
        self.bone_indices, self.bone_weights = sparsify_weights(body.skin_weights, BONE_INFLUENCES)
        self.posed = torch.zeros(body.num_bones, 1, device=device)
        posed = [n for n in POSED_BONES if config.pose_fingers or not n.startswith("finger")]
        self.posed[[body.bone(name) for name in posed]] = 1

        def weights(regressor: torch.Tensor) -> torch.Tensor:
            return skinned_point_weights(
                regressor, self.rest_vertices, self.bone_indices, self.bone_weights, body.num_bones
            )

        self.keypoint_weights = weights(body.keypoint_regressor.regression_weights)
        # (17,) metres from each keypoint to the nearest skin: how far behind the skin over it
        # the joint lies
        rest_keypoints = body.keypoint_regressor.regression_weights @ self.rest_vertices
        self.keypoint_insets = torch.cdist(rest_keypoints, self.rest_vertices).amin(dim=-1)
        self.extra_weights = None
        if observations.extra_keypoints is not None:
            vertices = surface_keypoint_vertices(body, self.rest_vertices, self.rest_bones)
            picked = torch.zeros(len(vertices), len(self.rest_vertices), device=device)
            picked[torch.arange(len(vertices)), vertices] = 1
            self.extra_weights = weights(picked)
        self.keypoint_importance = torch.ones(len(rest_keypoints), device=device)
        self.keypoint_importance[:5] = FACE_SHARE
        self.hand_weights = self._hand_weights(weights)
        self.leg_weights = self._joint_weights(
            [body.bone(f"{bone}.{side}") for side in "LR" for bone in LEG_BONES]
        )
        self.proxies = BodyProxies.build(body, self.rest_vertices)
        self.relaxed = relaxed_hands(body, self.rest_bones)
        self.fingers = torch.tensor(
            [[label.startswith("finger")] for label in body.bone_labels], device=device
        ).float()
        self.far_pairs = self.proxies.far_pairs(body)
        self.limits = JointLimits(body, self.rest_bones)
        self.samples = SurfaceSamples.draw(self.rest_vertices, body.faces, config.surface_samples)
        self.inset = silhouette_inset * self.samples.radius
        self.surfaces = observations.normals is not None
        if self.surfaces:
            import nvdiffrast.torch as dr

            self.raster = dr.RasterizeCudaContext()

    def tensor(self, array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.asarray(array)).float().to(self.device)

    def _joint_weights(self, bones: list[int]) -> torch.Tensor:
        """``(K, J, 4)`` point weights of the given bones' heads."""
        out = torch.zeros(len(bones), self.body.num_bones, 4, device=self.device)
        out[torch.arange(len(bones)), bones] = self.rest_bones[bones, :, 3]
        return out

    def _hand_weights(self, weights) -> torch.Tensor:
        """``(42, J, 4)`` point weights of ``HandJoints``' points, left hand first."""
        joints = HandJoints(self.body, self.rest_vertices, self.rest_bones)
        rows = []
        for side, bones in enumerate(joints.bones):
            rows.append(self._joint_weights([bones[0]]))
            for finger in range(5):
                rows.append(self._joint_weights(bones[1 + 3 * finger : 4 + 3 * finger]))
                tip = torch.zeros(1, len(self.rest_vertices), device=self.device)
                tip[0, joints.tips[5 * side + finger]] = 1
                rows.append(weights(tip))
        return torch.cat(rows)

    def start(self) -> tuple[np.ndarray, np.ndarray]:
        """``(P, N, J, 3)`` rotations and ``(P, N, 3)`` translations to start fitting from."""
        body = self.body
        candidates = self.tensor(orientations())
        turned = torch.zeros(len(candidates), body.num_bones, 3, device=self.device)
        turned[:, 0] = roma.rotmat_to_rotvec(candidates)
        skinning = body.skinning(self.rest_bones, turned, torch.zeros_like(turned[:, 0]))
        rest_keypoints = skinned_points(skinning, self.keypoint_weights)
        width, height = (int(v) for v in self.observations.image_size)
        camera = Camera.fixed(width, height, self.config.fov_y_degrees)
        root, translation = place_people(
            rest_keypoints.cpu().numpy(),
            self.observations.keypoints,
            camera.K.numpy().astype(np.float64),
            camera.view.numpy().astype(np.float64),
            MIN_KEYPOINT_CONFIDENCE,
        )
        rotations = np.zeros((*root.shape[:2], body.num_bones, 3), np.float32)
        rotations[:, :, 0] = roma.rotmat_to_rotvec(torch.from_numpy(root)).numpy()
        translation = translation.astype(np.float32)
        observations = self.observations
        for p in range(self.people):
            if observations.keypoint_depths is not None:
                # the skin's points at the estimate's scale, the joints behind them
                depths = observations.keypoint_depths[p]
                known = lifted_frames(depths)
                if known.any():
                    keypoints, depths = observations.keypoints[p, known], depths[known]
                    targets = self._world_points(keypoints[..., :2], depths, camera)
                    rays = self._world_points(keypoints[..., :2], np.ones_like(depths), camera)
                    seen = self.tensor(keypoints[..., 2] >= MIN_KEYPOINT_CONFIDENCE)[..., None]
                    posed, moved = pose_to_points(
                        body, self.rest_bones, self.posed, self.keypoint_weights, targets,
                        JOINT_STEPS, offsets=rays * self.keypoint_insets[:, None],
                        rays=torch.where(seen > 0, rays, torch.nan),
                        importance=self.keypoint_importance, limits=self.limits,
                    )  # fmt: skip
                    rotations[p, known] = posed.cpu().numpy()
                    translation[p, known] = moved.cpu().numpy()
                    logger.info(
                        "person %d: %d of %d frames start from keypoints lifted onto their depth",
                        p, known.sum(), len(known),
                    )  # fmt: skip
        # open, relaxed hands: held so, or where the fingers are posed, their start
        fingers = self.fingers[:, 0].bool().cpu().numpy()
        rotations[:, :, fingers] = self.relaxed.cpu().numpy()[fingers]
        return rotations, translation

    def _world_points(self, pixels: np.ndarray, depth: np.ndarray, camera: Camera) -> torch.Tensor:
        """``(n, K, 3)`` world points at ``(n, K)`` depths along the rays through ``(n, K, 2)``
        video pixels."""
        pixels, depth = self.tensor(pixels), self.tensor(depth)[..., None]
        K = camera.K.to(pixels)
        x = (pixels[..., 0:1] - K[0, 2]) / K[0, 0] * depth
        y = (pixels[..., 1:2] - K[1, 2]) / K[1, 1] * depth
        in_camera = torch.cat([x, y, depth], dim=-1)
        return in_camera @ camera.view[:3, :3].to(pixels)  # camera to world axes

    def targets(self, frames: slice) -> dict[str, torch.Tensor]:
        """The window's keypoints, hand points and masks, at the fitting size."""
        observations, config = self.observations, self.config
        keypoints = self.tensor(np.nan_to_num(observations.keypoints[:, frames]))
        masks = torch.from_numpy(observations.mask_frames(frames)).to(self.device)
        confidence = keypoints[..., 2] * (keypoints[..., 2] >= MIN_KEYPOINT_CONFIDENCE)
        if not self.independent:
            confidence = confidence * _occlusion_weights(
                keypoints[..., :2] * self.scale, masks, config.occluded_keypoint_weight
            )
        hands = self.tensor(np.nan_to_num(observations.hands[:, frames]))
        ignored = torch.from_numpy(observations.ignored_frames(frames)).to(self.device)
        compared = (~ignored).float() if self.independent else _unoccluded(masks.float()) * ~ignored
        targets = {
            "keypoints": keypoints[..., :2] * self.scale,
            "confidence": confidence,
            "hands": hands[..., :2] * self.scale,
            "hand_confidence": hands[..., 2] * (hands[..., 2] >= MIN_HAND_CONFIDENCE),
            "masks": masks.float(),
            "compared": compared,
        }
        if self.extra_weights is not None:
            extra = self.tensor(np.nan_to_num(observations.extra_keypoints[:, frames]))
            targets["extra"] = extra[..., :2] * self.scale
            targets["extra_confidence"] = extra[..., 2] * (extra[..., 2] >= MIN_KEYPOINT_CONFIDENCE)
        if observations.keypoint_depths is not None:
            targets["keypoint_depths"] = self.tensor(observations.keypoint_depths[:, frames])
        if self.surfaces:
            normals = self.tensor(observations.normals[:, frames])
            inner = -torch.nn.functional.max_pool2d(
                -masks.float().flatten(0, 1)[:, None],
                2 * SURFACE_EDGE + 1,
                stride=1,
                padding=SURFACE_EDGE,
            )[:, 0].unflatten(0, masks.shape[:2])
            targets["normals"] = normals
            targets["depths"] = self.tensor(observations.depths[:, frames])
            targets["surface_region"] = inner * compared * (normals.norm(dim=-1) > 0.5)
        return targets

    def skinning(self, params: _MotionParameters, frames: slice = slice(None)) -> torch.Tensor:
        """``(P, n, J, 4, 4)`` skinning matrices of the window's frames."""
        rotations = params.rotations(frames)
        skinning = self.body.skinning(
            self.rest_bones, rotations.flatten(0, 1), params.translation[:, frames].flatten(0, 1)
        )
        return skinning.unflatten(0, rotations.shape[:2])

    def vertices(self, skinning: torch.Tensor) -> torch.Tensor:
        """``(P * n, V, 3)`` skinned vertices for ``(P, n, J, 4, 4)`` skinning matrices."""
        return linear_blend_skinning(
            self.rest_vertices, self.bone_indices, self.bone_weights, skinning.flatten(0, 1)
        )

    def silhouettes(self, skinning: torch.Tensor, vertices: torch.Tensor | None = None):
        """``(P, n, h, w)`` rendered silhouettes for ``(P, n, J, 4, 4)`` skinning matrices
        (whose ``vertices`` may be given)."""
        vertices = self.vertices(skinning) if vertices is None else vertices
        points = self.samples.points(vertices, self.body.faces, self.inset)
        return render_silhouette(points, self.samples.radius, self.camera).unflatten(
            0, skinning.shape[:2]
        )

    def surface_terms(
        self, vertices: torch.Tensor, targets: dict[str, torch.Tensor], chunk: slice
    ) -> dict[str, torch.Tensor]:
        """The normal and depth terms (unweighted) of ``(P * n, V, 3)`` vertices of a chunk."""
        drawn = render_surface(self.raster, vertices, self.body.faces, self.camera, NEAR, FAR)
        region = targets["surface_region"][:, chunk].flatten(0, 1) * drawn.covered
        normals = targets["normals"][:, chunk].flatten(0, 1)
        depths = targets["depths"][:, chunk].flatten(0, 1)
        return {
            "normals": normal_loss(drawn.normals, normals, region),
            "depth": sum(
                depth_loss(drawn.depth, depths[:, k], region) for k in range(depths.shape[1])
            )
            / depths.shape[1],
        }

    def terms(
        self, params: _MotionParameters, targets: dict[str, torch.Tensor], phase: int
    ) -> dict[str, torch.Tensor]:
        """Every term of the loss but the silhouettes', over the whole window."""
        config, height = self.config, self.camera.height
        skinning = self.skinning(params)
        screen = self.camera.screen(skinned_points(skinning, self.keypoint_weights))
        keypoints = screen[..., :2]
        hip_l, knee_l, ankle_l, hip_r, knee_r, ankle_r = skinned_points(
            skinning, self.leg_weights
        ).unbind(-2)
        lateral = hip_l - hip_r
        root = params.root_rotation().transpose(0, 1).flatten(-2)  # frames first
        pose = (params.pose * params.posed).transpose(0, 1)
        translation = params.translation.transpose(0, 1)
        terms = {
            "keypoints": config.keypoint_weight
            * keypoint_loss(keypoints, targets["keypoints"], targets["confidence"], height),
            "knees": config.knee_weight
            * (
                knee_bend(hip_l, knee_l, ankle_l, lateral)
                + knee_bend(hip_r, knee_r, ankle_r, lateral)
            ).mean(),
            "smoothness": config.smoothness_weight
            * (
                second_difference(root).square().sum(-1).mean()
                + second_difference(pose).square().sum(-1).sum(-1).mean()
                + 100 * second_difference(translation).square().sum(-1).mean()
            ),
            "pose_prior": config.pose_prior_weight
            * (pose * (1 - self.fingers)).square().sum(-1).sum(-1).mean(),
            "joint_limits": config.joint_limit_weight * self.limits(pose).mean(),
        }
        if config.pose_fingers:  # unsure hand points leave the fingers relaxed
            terms["finger_prior"] = (
                config.finger_prior_weight
                * ((pose - self.relaxed) * self.fingers).square().sum(-1).sum(-1).mean()
            )
        if "extra" in targets:
            terms["extra_keypoints"] = config.keypoint_weight * keypoint_loss(
                self.camera.project(skinned_points(skinning, self.extra_weights)),
                targets["extra"],
                targets["extra_confidence"],
                height,
            )
        if "keypoint_depths" in targets:
            terms["keypoint_depth"] = config.keypoint_depth_weight * keypoint_depth_loss(
                screen[..., 2],
                targets["keypoint_depths"],
                self.keypoint_insets,
                self.keypoint_importance,
            )
        spheres = self.proxies.place(skinning)
        terms["self_contact"] = config.self_contact_weight * self_overlap(
            spheres, self.far_pairs, config.contact_slack
        )
        if self.people > 1 and not self.independent:
            terms["contact"] = config.contact_weight * overlap(spheres, config.contact_slack)
        if phase >= 1:
            hands = self.camera.project(skinned_points(skinning, self.hand_weights))
            hands = hands.unflatten(-2, (2, -1))
            terms["hands"] = config.hand_weight * keypoint_loss(
                hands, targets["hands"], targets["hand_confidence"], height
            )
        return terms

    def fit_window(
        self, rotations: torch.Tensor, translation: torch.Tensor, targets: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, list[int], dict[str, float]]:
        """Fit one window from ``(P, n, J, 3)`` rotations and ``(P, n, 3)`` translations.

        Returns:
            The fitted rotations and translations, the steps each phase took, and the last
            values of the loss terms.
        """
        config = self.config
        params = _MotionParameters(self.posed, rotations, translation)
        frames = rotations.shape[1]
        chunks = [
            slice(start, min(start + config.chunk_frames, frames))
            for start in range(0, frames, config.chunk_frames)
        ]
        steps, last = [], {}
        for phase, max_steps in enumerate(config.iterations):
            optimizer = torch.optim.Adam(params.parameters(), lr=config.learning_rate)
            previous = None
            step = 0
            for step in range(1, max_steps + 1):
                optimizer.zero_grad(set_to_none=True)
                terms = self.terms(params, targets, phase)
                surfaces = self.surfaces and phase >= 1
                if phase >= 2 or surfaces:
                    # rendered a chunk at a time, each chunk's gradients right away
                    rendered_terms = {}
                    weights = {
                        "silhouette": config.silhouette_weight,
                        "normals": config.normal_weight,
                        "depth": config.depth_weight,
                    }
                    for chunk in chunks:
                        skinning = self.skinning(params, chunk)
                        vertices = self.vertices(skinning)
                        parts = {}
                        if phase >= 2:
                            parts["silhouette"] = silhouette_loss(
                                self.silhouettes(skinning, vertices),
                                targets["masks"][:, chunk],
                                targets["compared"][:, chunk],
                            )
                        if surfaces:
                            parts.update(self.surface_terms(vertices, targets, chunk))
                        share = (chunk.stop - chunk.start) / frames
                        sum(weights[k] * share * v for k, v in parts.items()).backward()
                        for k, v in parts.items():
                            rendered_terms[k] = rendered_terms.get(k, 0) + share * v.detach()
                    rendered = {k: weights[k] * v for k, v in rendered_terms.items()}
                    terms.update(rendered)
                else:
                    rendered = {}
                sum(value for name, value in terms.items() if name not in rendered).backward()
                optimizer.step()
                if step % config.check_every == 0 or step == max_steps:
                    total = sum(value.detach() for value in terms.values()).item()
                    if previous is not None and previous - total < config.tolerance * previous:
                        break
                    previous = total
            steps.append(step)
            last = {name: value.item() for name, value in terms.items()}
        return params.rotations().detach(), params.translation.detach(), steps, last

    @torch.no_grad()
    def evaluate(
        self, rotations: torch.Tensor, translation: torch.Tensor, targets: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(P, n)`` keypoint errors in pixels and silhouette IoUs of fitted frames."""
        skinning = self.body.skinning(
            self.rest_bones, rotations.flatten(0, 1), translation.flatten(0, 1)
        ).unflatten(0, rotations.shape[:2])
        projected = self.camera.project(skinned_points(skinning, self.keypoint_weights))
        distance = (projected - targets["keypoints"]).norm(dim=-1) / self.scale
        seen = targets["confidence"] > 0
        errors = (distance * seen).sum(-1) / seen.sum(-1).clamp_min(1)
        ious = []
        for start in range(0, rotations.shape[1], self.config.chunk_frames):
            chunk = slice(start, start + self.config.chunk_frames)
            masks = targets["masks"][:, chunk]
            region = targets["compared"][:, chunk] > 0.5
            rendered = self.silhouettes(skinning[:, chunk])
            ious.append(
                silhouette_iou(
                    (rendered * region).flatten(0, 1), (masks * region).flatten(0, 1)
                ).unflatten(0, masks.shape[:2])
            )
        return errors, torch.cat(ious, dim=1)


def fit_motion(
    observations: Observations,
    body: AnnyBody,
    shape: BodyFit,
    expression_names: list[str],
    config: MotionConfig,
    silhouette_inset: float,
    independent: bool = False,
) -> MotionFit:
    """Fit the avatar's body, shaped as in ``shape``, to every person in a motion video.

    Args:
        expression_names: the avatar's expressions, which the faces' blendshapes are mapped to.
        silhouette_inset: as ``FittingConfig.silhouette_inset``.
        independent: the "people" are not in one scene but separate stretches fitted together
            (one step for all is faster than one fit each): they neither touch nor hide each
            other, so contact and occlusion are left out.
    """
    fitter = _Fitter(observations, body, shape, config, silhouette_inset, independent)
    rotations, translation = fitter.start()
    people, frames = translation.shape[:2]
    errors = np.zeros((people, frames), np.float32)
    ious = np.zeros((people, frames), np.float32)
    windows = _windows(frames, config.window_frames, config.window_overlap)
    done = 0
    for index, window in enumerate(windows):
        targets = fitter.targets(window)
        fitted_rotations, fitted_translation, steps, terms = fitter.fit_window(
            fitter.tensor(rotations[:, window]), fitter.tensor(translation[:, window]), targets
        )
        error, iou = fitter.evaluate(fitted_rotations, fitted_translation, targets)
        blend = _blend_weights(window, done)
        rotations[:, window] = _blend_rotations(
            rotations[:, window], fitted_rotations.cpu().numpy(), blend
        )
        translation[:, window] += blend[:, None] * (
            fitted_translation.cpu().numpy() - translation[:, window]
        )
        errors[:, window] += blend * (error.cpu().numpy() - errors[:, window])
        ious[:, window] += blend * (iou.cpu().numpy() - ious[:, window])
        done = window.stop
        logger.info(
            "window %d of %d, frames %d to %d: %s steps; %s",
            index + 1, len(windows), window.start, window.stop - 1, "/".join(map(str, steps)),
            ", ".join(f"{k} {v:.2e}" for k, v in terms.items()),
        )  # fmt: skip

    confidence = np.nan_to_num(observations.keypoints[..., 2])
    return MotionFit(
        frame_indices=observations.frame_indices,
        fps=observations.fps,
        fov_y_degrees=config.fov_y_degrees,
        image_size=observations.image_size,
        phenotype=shape.phenotype,
        local_changes=shape.local_changes,
        rotations=rotations,
        translation=translation,
        expressions=face_expressions(
            observations.blendshapes,
            list(observations.blendshape_names),
            expression_names,
            config.expression_smoothing,
        ),
        expression_names=np.array(expression_names),
        seen=(confidence >= MIN_KEYPOINT_CONFIDENCE).sum(-1) >= 4,
        keypoint_error=errors,
        silhouette_iou=ious,
    )


JOINT_STEPS = 300
"""Steps posing the body onto its keypoints lifted to 3D."""
MIN_LIFTED = 4
"""Body keypoints (shoulders to ankles) lifted onto their estimated depth a frame needs to
start from them: the face's alone say little of how the body lies."""
LIMIT_WEIGHT = 0.1
"""Weight of the joint limits, per squared radian past them, against squared metres missed when
posing onto 3D points: 10 cm missed weigh as much as 18 degrees past a limit."""
OUTLIER_FLOOR = 0.05
"""Metres: lifted keypoints missed by less than this are never taken for hidden ones."""
FACE_SHARE = 0.2
"""Weight of each of the five face keypoints in the depth terms: together they count as one,
not as five points at nearly one depth that would set the body's turn."""


def lifted_frames(keypoint_depths: np.ndarray) -> np.ndarray:
    """``(...)`` bool: which frames of ``(..., 17)`` ``Observations.keypoint_depths`` the motion
    fit starts from them."""
    return np.isfinite(keypoint_depths[..., 5:]).sum(-1) >= MIN_LIFTED


def pose_to_points(
    body: AnnyBody,
    rest_bones: torch.Tensor,
    posed: torch.Tensor,
    point_weights: torch.Tensor,
    targets: torch.Tensor,
    steps: int,
    offsets: torch.Tensor | None = None,
    rays: torch.Tensor | None = None,
    importance: torch.Tensor | None = None,
    limits: JointLimits | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pose a body so its points (bone heads, keypoints, ...) lie on ``(n, K, 3)`` world points
    (NaN: unknown).

    The root is turned and moved rigidly onto the points first (Kabsch, on the known ones),
    then every bone with ``(J, 1)`` ``posed`` set turns to bring its points onto theirs.

    Args:
        point_weights: ``(K, J, 4)`` the points' weights (``skinned_point_weights``).
        offsets: ``(n, K, 3)`` where the targets are an estimate of the skin over the points,
            at the estimate's own scale: the targets are scaled about the world's origin, the
            camera, by a factor fitted per frame (so they keep their pixels), and these offsets
            (from the skin to the point behind it) added after. Such a target may also be the
            skin of whatever hides the point, so halfway the points that miss theirs by far
            are held to them only one way: no nearer the camera than their target.
        rays: ``(n, K, 3)`` directions from the camera (the world's origin) through the points'
            pixels, NaN where unseen: points seen but without a target are kept on their ray.
        importance: ``(K,)`` each point's weight; 1 by default.
        limits: where given, the bones are kept within them.

    Returns:
        ``(n, J, 3)`` rotation vectors and ``(n, 3)`` translations.
    """
    known = torch.isfinite(targets).all(dim=-1)  # (n, K)
    targets = torch.nan_to_num(targets)
    if importance is None:
        importance = torch.ones(targets.shape[1], device=targets.device)
    if rays is None:
        rays = torch.full_like(targets, torch.nan)
    on_rays = (torch.isfinite(rays).all(dim=-1) & ~known) * importance
    rays = torch.nn.functional.normalize(torch.nan_to_num(rays), dim=-1)
    directions = torch.nn.functional.normalize(targets, dim=-1)  # from the camera
    zeros = torch.zeros(1, body.num_bones, 3, device=targets.device)
    rest_points = skinned_points(body.skinning(rest_bones, zeros, zeros[:, 0]), point_weights)[0]
    weights = known * importance
    total = weights.sum(dim=1, keepdim=True).clamp_min(1e-6)

    def centred(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        centre = (weights[..., None] * points).sum(dim=1, keepdim=True) / total[..., None]
        return centre[:, 0], points - centre

    rest_centre, rest_spread = centred(rest_points.expand_as(targets))
    log_scale = torch.zeros(len(targets), device=targets.device)
    if offsets is not None:  # scaled to the body's spread to start
        spread = (weights * centred(targets)[1].square().sum(-1)).sum(-1)
        log_scale = 0.5 * torch.log(
            (weights * rest_spread.square().sum(-1)).sum(-1) / spread.clamp_min(1e-9)
        )
        offsets = torch.nan_to_num(offsets)
    log_scale = torch.nn.Parameter(log_scale, requires_grad=offsets is not None)

    def placed_targets() -> torch.Tensor:
        if offsets is None:
            return targets
        return log_scale.exp()[:, None, None] * targets + offsets

    with torch.no_grad():
        target_centre, target_spread = centred(placed_targets())
    covariance = torch.einsum("nk,nka,nkb->nab", weights, target_spread, rest_spread)
    u, _, vt = torch.linalg.svd(covariance)
    d = torch.ones_like(u[:, 0])
    d[:, 2] = torch.sign(torch.linalg.det(u @ vt))
    rotation = u @ torch.diag_embed(d) @ vt  # (n, 3, 3): rest to target
    root_head = rest_bones[0, :3, 3]
    # skinning turns about the root's head: x -> R (x - h) + h + t
    shift = target_centre - torch.einsum("nab,nb->na", rotation, rest_centre - root_head)
    shift = shift - root_head

    count = len(targets)
    root = torch.nn.Parameter(roma.rotmat_to_rotvec(rotation))
    pose = torch.nn.Parameter(torch.zeros(count, body.num_bones, 3, device=targets.device))
    with torch.no_grad():  # the zero pose is not the rest pose: what that shifts, moved back
        skinning = body.skinning(rest_bones, torch.cat([root[:, None], pose[:, 1:]], 1), shift)
        shift += target_centre - centred(skinned_points(skinning, point_weights))[0]
    translation = torch.nn.Parameter(shift)
    optimizer = torch.optim.Adam([root, pose, translation, log_scale], lr=0.01)
    one_way = torch.zeros_like(weights)  # weights of the points held only one way

    def rotations() -> torch.Tensor:
        return torch.cat([root[:, None], (pose * posed)[:, 1:]], dim=1)

    for step in range(steps):
        skinning = body.skinning(rest_bones, rotations(), translation)
        placed = skinned_points(skinning, point_weights)
        goal = placed_targets()
        off_target = (placed - goal).square().sum(-1)
        if offsets is not None and step == steps // 2:
            with torch.no_grad():
                miss = off_target.sqrt()
                median = torch.where(known, miss, torch.nan).nanmedian(dim=1, keepdim=True).values
                far = known & (miss > (3 * median).clamp_min(OUTLIER_FLOOR))
                one_way = torch.where(far, weights, 0)
                weights = torch.where(far, 0, weights)
        nearer = torch.relu(((goal - placed) * directions).sum(-1)).square()
        off_ray = (placed - (placed * rays).sum(-1, keepdim=True) * rays).square().sum(-1)
        error = (off_target * weights + nearer * one_way + off_ray * on_rays).sum() / (
            weights + one_way + on_rays
        ).sum().clamp_min(1e-6)
        loss = error + 1e-4 * (pose * posed).square().sum(-1).mean()
        if limits is not None:
            loss = loss + LIMIT_WEIGHT * limits(pose * posed).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return rotations().detach(), translation.detach()


def _windows(frames: int, size: int, overlap: int) -> list[slice]:
    """Windows of ``size`` frames covering ``frames``, each overlapping the one before by
    ``overlap``; the last ends at the last frame."""
    if frames <= size:
        return [slice(0, frames)]
    stride = size - overlap
    starts = list(range(0, frames - size, stride)) + [frames - size]
    return [slice(start, start + size) for start in starts]


def _blend_weights(window: slice, done: int) -> np.ndarray:
    """``(n,)`` how much a window's result counts per frame: rising linearly across the frames
    an earlier window already fitted, fully from there on."""
    frames = np.arange(window.start, window.stop)
    overlap = max(done - window.start, 0)
    return np.clip((frames - window.start + 1) / (overlap + 1), 0, 1).astype(np.float32)


def _blend_rotations(old: np.ndarray, new: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """``(P, n, J, 3)`` rotation vectors interpolated per frame from ``old`` to ``new`` by
    ``(n,)`` weights, along the shorter way (normalized quaternion interpolation)."""
    a = roma.rotvec_to_unitquat(torch.from_numpy(old).double())
    b = roma.rotvec_to_unitquat(torch.from_numpy(new).double())
    b = torch.where((a * b).sum(-1, keepdim=True) < 0, -b, b)
    t = torch.from_numpy(weights).double()[None, :, None, None]
    mixed = torch.nn.functional.normalize(a + t * (b - a), dim=-1)
    return roma.unitquat_to_rotvec(mixed).float().numpy()


def _occlusion_weights(keypoints: torch.Tensor, masks: torch.Tensor, weight: float) -> torch.Tensor:
    """``(P, n, 17)`` factor per keypoint: ``weight`` where it lies on another person's mask and
    not on its own, 1 elsewhere.

    Args:
        keypoints: ``(P, n, 17, 2)`` pixels at the masks' size.
        masks: ``(P, n, h, w)`` bool.
    """
    people, frames = keypoints.shape[:2]
    factors = torch.ones(keypoints.shape[:3], device=keypoints.device)
    if people < 2:
        return factors
    height, width = masks.shape[2:]
    x = keypoints[..., 0].round().long().clamp(0, width - 1)
    y = keypoints[..., 1].round().long().clamp(0, height - 1)
    frame = torch.arange(frames, device=keypoints.device)[:, None]
    # on[p, q]: whether person p's keypoints lie on person q's mask.
    on = torch.stack(
        [torch.stack([masks[q][frame, y[p], x[p]] for q in range(people)]) for p in range(people)]
    )
    for p in range(people):
        others = torch.stack([on[p, q] for q in range(people) if q != p]).any(dim=0)
        factors[p][others & ~on[p, p]] = weight
    return factors


def _unoccluded(masks: torch.Tensor) -> torch.Tensor:
    """``(P, n, h, w)`` where each person can be compared: not inside someone else's mask."""
    if len(masks) < 2:
        return torch.ones_like(masks)
    total = masks.sum(dim=0, keepdim=True)
    return 1 - (total - masks).clamp(0, 1)


def expression_mapping(blendshape_names: list[str], expression_names: list[str]) -> np.ndarray:
    """``(B, E)`` matrix taking MediaPipe's ARKit blendshape weights to the avatar's expressions.

    ARKit names a side ``Left``/``Right`` where ICT-FaceKit writes ``_L``/``_R``; ICT-FaceKit
    splits a few of ARKit's two-sided shapes (``browInnerUp``, ``cheekPuff``), which drive both
    of its halves. ``jawLeft``, ``mouthRight`` and the like name a direction and keep their name.
    """
    index = {name: i for i, name in enumerate(expression_names)}
    mapping = np.zeros((len(blendshape_names), len(expression_names)), np.float32)
    for b, name in enumerate(blendshape_names):
        if name in index:
            mapping[b, index[name]] = 1
            continue
        for side, suffix in (("Left", "_L"), ("Right", "_R")):
            if name.endswith(side) and name.removesuffix(side) + suffix in index:
                mapping[b, index[name.removesuffix(side) + suffix]] = 1
                break
        else:
            for suffix in ("_L", "_R"):
                if name + suffix in index:
                    mapping[b, index[name + suffix]] = 1
    return mapping


def face_expressions(
    blendshapes: np.ndarray,
    blendshape_names: list[str],
    expression_names: list[str],
    smoothing: int,
) -> np.ndarray:
    """``(P, N, E)`` the avatar's expressions from ``(P, N, B)`` blendshape weights.

    Frames without a face take their neighbours' weights, or a neutral face where no frame has
    one; a median filter over ``smoothing`` frames removes the landmarker's flicker.
    """
    people, frames = blendshapes.shape[:2]
    out = np.zeros((people, frames, len(expression_names)), np.float32)
    if not blendshape_names:
        return out
    mapping = expression_mapping(blendshape_names, expression_names)
    for p in range(people):
        found = np.flatnonzero(~np.isnan(blendshapes[p]).any(axis=1))
        if not found.size:
            continue
        filled = np.stack(
            [
                np.interp(np.arange(frames), found, blendshapes[p, found, b])
                for b in range(len(mapping))
            ],
            axis=1,
        )
        if smoothing > 1:
            filled = scipy.ndimage.median_filter(filled, size=(smoothing, 1), mode="nearest")
        out[p] = np.clip(filled @ mapping, 0, 1)
    return out


@torch.no_grad()
def fitted_people(fit: MotionFit, body: AnnyBody, frames: list[int]) -> torch.Tensor:
    """``(P, n, V, 3)`` the fitted bodies' vertices in the given frames."""
    device = body.faces.device

    def tensor(array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.asarray(array)).float().to(device)

    rotations = tensor(fit.rotations[:, frames])
    translation = tensor(fit.translation[:, frames])
    _, vertices = pose_body(
        body,
        tensor(fit.phenotype)[None],
        tensor(fit.local_changes)[None],
        rotations.flatten(0, 1),
        translation.flatten(0, 1),
    )
    return vertices.unflatten(0, rotations.shape[:2])
