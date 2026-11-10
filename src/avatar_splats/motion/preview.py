"""Playing a clip on the exported avatar with the reference runtime, to watch it against the
video it came from."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import roma
import torch

from avatar_splats.cameras import Camera
from avatar_splats.export.runtime import AvatarRuntime
from avatar_splats.motion.clip import MotionClip


def clip_camera(clip: MotionClip, camera: Camera) -> Camera:
    """``camera``'s lens and image size, placed where the video's camera was in the clip."""
    view = torch.linalg.inv(torch.from_numpy(clip.camera_to_world)).float().to(camera.K.device)
    return Camera(camera.K, view, camera.width, camera.height)


def pose_clip(runtime: AvatarRuntime, clip: MotionClip, frame: int) -> torch.Tensor:
    """``(P, V, 3)`` a copy of the avatar per person in the clip, posed as in ``frame``.

    Bones and expressions are matched by name, so a clip plays on any avatar with Anny's
    skeleton and the ICT-FaceKit expressions.
    """
    device = runtime.rig.bind_poses.device
    bones = [list(clip.bone_names).index(name) for name in runtime.bone_names]
    names = list(clip.expression_names)
    expressions = np.zeros((clip.num_people, len(runtime.expression_names)), np.float32)
    for i, name in enumerate(runtime.expression_names):
        if name in names:
            expressions[:, i] = clip.expressions[:, frame, names.index(name)]
    wxyz = torch.from_numpy(clip.rotations[:, frame, bones]).to(device)
    rotations = roma.unitquat_to_rotmat(wxyz[..., [1, 2, 3, 0]])
    return runtime.pose(
        rotations,
        torch.from_numpy(expressions).to(device),
        torch.from_numpy(clip.root_translation[:, frame]).to(device),
    )


@torch.no_grad()
def render_clip(
    runtime: AvatarRuntime,
    clip: MotionClip,
    camera: Camera,
    background: torch.Tensor,
    frames: range | None = None,
) -> Iterator[np.ndarray]:
    """``(H, W, 3)`` uint8 images of everyone in the clip, frame by frame, seen by ``camera``."""
    for frame in frames if frames is not None else range(clip.rotations.shape[1]):
        image = runtime.render(pose_clip(runtime, clip, frame), camera, background)
        yield (image.clamp(0, 1) * 255).byte().cpu().numpy()
