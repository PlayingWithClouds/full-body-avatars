"""Animation clips: fitted motion as the runtime plays it.

A clip holds, per person and frame, each bone's rotation relative to its bind pose (as
``export/runtime.py`` and the viewer expect), the root's offset from its bind position, and the
face's expression weights. Its world is levelled: the video's camera need not have been level, so
the floor the people stand or lie on is found and turned to z = 0, and the people are centred on
the origin at the clip's first frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import roma
import torch

from avatar_splats.body_models.anny import AnnyBody
from avatar_splats.body_models.hybrid import Avatar
from avatar_splats.export.avatar_file import AvatarFile, write_avatar
from avatar_splats.fitting.body import pose_body
from avatar_splats.motion.fit import MotionFit
from avatar_splats.records import ArrayRecord

MOTION_MAGIC = b"MOTION01"
"""Magic of ``.motion`` files: the ``.avatar`` layout (``export/avatar_file.py``) holding a clip."""
FLOOR_TRIM = 0.5
"""Share of the lowest points kept when fitting the floor: people jump, kneel, and lift feet."""
FLOOR_ITERATIONS = 3
LEVEL_SPREAD = 0.1
"""Holds the floor level unless the lowest points spread out over much more than this, in
metres: a person standing on one spot tells nothing of the floor's tilt."""


@dataclass(frozen=True)
class MotionClip(ArrayRecord):
    """Bind-relative animation of one or more copies of the avatar."""

    fps: float
    bone_names: np.ndarray
    """``(J,)`` the avatar's bones."""
    expression_names: np.ndarray
    """``(E,)`` the avatar's expressions."""
    rotations: np.ndarray
    """``(P, N, J, 4)`` wxyz quaternion per person, frame and bone, relative to its bind pose."""
    root_translation: np.ndarray
    """``(P, N, 3)`` the root's offset from its bind position, in metres in world axes."""
    expressions: np.ndarray
    """``(P, N, E)`` expression weights in [0, 1]."""
    seen: np.ndarray
    """``(P, N)`` bool: whether the person was seen in the frame or only interpolated."""
    camera_to_world: np.ndarray
    """``(4, 4)`` where the video's camera was in the clip's world (OpenCV axes), to view the
    clip as the video did."""
    fov_y_degrees: float
    """The video camera's vertical field of view."""

    @property
    def num_people(self) -> int:
        return len(self.rotations)


def bind_relative_pose(
    bone_poses: torch.Tensor, bind_poses: torch.Tensor, parents: list[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """The runtime's pose for posed bone frames: the inverse of ``runtime.forward_kinematics``.

    Each bone sits on its parent as in the bind pose and turns about its own frame; the root
    also moves, by an offset in world axes.

    Args:
        bone_poses: ``(..., J, 4, 4)`` posed bone frames in the world; rigid.
        bind_poses: ``(J, 4, 4)`` bone frames in the bind pose.
        parents: parent per bone, -1 for the root (bone 0).

    Returns:
        ``(..., J, 3, 3)`` rotations in each bone's bind frame and ``(..., 3)`` the root's offset.
    """
    inverse_bind = _invert_rigid(bind_poses)
    parent = torch.tensor([max(p, 0) for p in parents], device=bind_poses.device)
    relative = inverse_bind[parent] @ bind_poses
    relative[0] = bind_poses[0]
    base = bone_poses[..., parent, :, :] @ relative
    base[..., 0, :, :] = bind_poses[0]
    rotations = (_invert_rigid(base) @ bone_poses)[..., :3, :3]
    offset = bone_poses[..., 0, :3, 3] - bind_poses[0, :3, 3]
    return rotations, offset


def floor_transform(points: np.ndarray) -> np.ndarray:
    """``(4, 4)`` the rigid transform turning the floor under ``(M, 3)`` lowest points to z = 0.

    The floor is a plane z = a x + b y + c fitted to the lowest points, refitted to the lowest
    ``FLOOR_TRIM`` of them a few times; its tilt is held near level unless the points spread
    out enough to tell it.
    """
    kept = points
    a = b = c = 0.0
    for _ in range(FLOOR_ITERATIONS):
        centre = kept.mean(axis=0)
        design = kept[:, :2] - centre[:2]
        ridge = len(kept) * LEVEL_SPREAD**2 * np.eye(2)
        a, b = np.linalg.solve(design.T @ design + ridge, design.T @ (kept[:, 2] - centre[2]))
        c = centre[2] - a * centre[0] - b * centre[1]
        residual = points[:, 2] - (a * points[:, 0] + b * points[:, 1] + c)
        kept = points[residual <= np.quantile(residual, FLOOR_TRIM)]
    normal = np.array([-a, -b, 1.0]) / np.linalg.norm([-a, -b, 1.0])
    rotation = roma.rotvec_to_rotmat(
        torch.from_numpy(_rotvec_between(normal, np.array([0.0, 0, 1])))
    ).numpy()
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[2, 3] = -(rotation @ np.array([0.0, 0, c]))[2]
    return transform


def motion_clip(fit: MotionFit, body: AnnyBody, avatar: Avatar, chunk: int = 64) -> MotionClip:
    """The clip of a fitted motion video, levelled and centred."""
    device = body.faces.device

    def tensor(array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.asarray(array)).float().to(device)

    people, frames, bones = fit.rotations.shape[:3]
    phenotype, local_changes = tensor(fit.phenotype)[None], tensor(fit.local_changes)[None]
    rotations = tensor(fit.rotations).flatten(0, 1)
    translation = tensor(fit.translation).flatten(0, 1)
    bind_poses = tensor(avatar.bind_poses)
    poses, lowest = [], []
    with torch.no_grad():
        for start in range(0, len(rotations), chunk):
            part = slice(start, start + chunk)
            skinning, vertices = pose_body(
                body, phenotype, local_changes, rotations[part], translation[part]
            )
            poses.append(skinning @ bind_poses)
            lowest.append(vertices[torch.arange(len(vertices)), vertices[..., 2].argmin(dim=1)])
    bone_poses = torch.cat(poses).double()
    lowest = torch.cat(lowest).cpu().numpy()

    level = floor_transform(lowest[fit.seen.reshape(-1)] if fit.seen.any() else lowest)
    roots = (level[:3, :3] @ bone_poses[:, 0, :3, 3].cpu().numpy().T).T + level[:3, 3]
    level[:2, 3] -= roots.reshape(people, frames, 3)[:, 0, :2].mean(axis=0)
    bone_poses = tensor(level).double() @ bone_poses

    bind_rotations, offsets = bind_relative_pose(
        bone_poses, bind_poses.double(), avatar.bone_parents.tolist()
    )
    quaternions = roma.rotmat_to_unitquat(bind_rotations)[..., [3, 0, 1, 2]]  # xyzw to wxyz
    return MotionClip(
        fps=fit.fps,
        bone_names=avatar.bone_labels,
        expression_names=fit.expression_names,
        rotations=quaternions.unflatten(0, (people, frames)).float().cpu().numpy(),
        root_translation=offsets.unflatten(0, (people, frames)).float().cpu().numpy(),
        expressions=fit.expressions,
        seen=fit.seen,
        camera_to_world=level @ np.linalg.inv(fit.camera().view.double().numpy()),
        fov_y_degrees=fit.fov_y_degrees,
    )


def write_motion(path: str | Path, clip: MotionClip) -> None:
    """Write a clip as a ``.motion`` file for the viewer and the game runtime.

    The metadata gives ``fps``, ``frames``, ``people``, the ``bones`` and ``expressions`` the
    arrays are ordered by, and the video camera's ``fov_y_degrees``. The arrays are
    ``rotations`` ``(P, N, J, 4)`` wxyz bind-relative quaternions, ``root_translation``
    ``(P, N, 3)`` metres, ``expressions`` ``(P, N, E)``, ``seen`` ``(P, N)`` uint8 and
    ``camera_to_world`` ``(4, 4)``, all float32 but ``seen``.
    """
    people, frames = clip.rotations.shape[:2]
    metadata = {
        "format_version": 1,
        "units": "metres",
        "up_axis": "+z",
        "forward_axis": "-y",
        "quaternion_order": "wxyz",
        "fps": float(clip.fps),
        "frames": int(frames),
        "people": int(people),
        "bones": [str(name) for name in clip.bone_names],
        "expressions": [str(name) for name in clip.expression_names],
        "fov_y_degrees": float(clip.fov_y_degrees),
    }
    arrays = {
        "rotations": clip.rotations.astype(np.float32),
        "root_translation": clip.root_translation.astype(np.float32),
        "expressions": clip.expressions.astype(np.float32),
        "seen": clip.seen.astype(np.uint8),
        "camera_to_world": clip.camera_to_world.astype(np.float32),
    }
    write_avatar(path, AvatarFile(metadata, arrays), MOTION_MAGIC)


def _invert_rigid(transforms: torch.Tensor) -> torch.Tensor:
    inverse = transforms.clone()
    rotation = transforms[..., :3, :3].transpose(-1, -2)
    inverse[..., :3, :3] = rotation
    inverse[..., :3, 3] = -(rotation @ transforms[..., :3, 3:])[..., 0]
    return inverse


def _rotvec_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``(3,)`` the shortest rotation taking unit vector ``a`` to unit vector ``b``."""
    axis = np.cross(a, b)
    sine = np.linalg.norm(axis)
    if sine < 1e-12:
        return np.zeros(3)
    return axis / sine * np.arctan2(sine, np.dot(a, b))
