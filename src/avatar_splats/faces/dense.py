"""Where MediaPipe's face-mesh landmarks lie on the head model.

The 68 landmarks the head fit uses are a sparse sample; a grin, a wrinkled nose or a frown moves
the cheeks, nostrils and brows between them. MediaPipe's 468 mesh landmarks follow all of that,
but are not defined on ICT-FaceKit's mesh. Their counterparts are found once from videos whose
head fits are good: for each landmark, the skin vertex that, posed as the fit poses it, lands
nearest to the landmark over all frames.
"""

from __future__ import annotations

import numpy as np
import torch

from avatar_splats.body_models import ict
from avatar_splats.body_models.ict import IctFaceKit
from avatar_splats.fitting.head import HeadFit, head_vertices

NUM_MESH_LANDMARKS = 468
"""MediaPipe's face mesh; the ten after it are the irises."""
FACE_OVAL = (
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
    152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
)  # fmt: skip
"""The mesh landmarks along the face's outline, which slides over the head as it turns."""


@torch.no_grad()
def landmark_vertices(
    model: IctFaceKit,
    videos: list[tuple[HeadFit, np.ndarray]],
    max_yaw: float,
    cap: float,
    device: str,
    chunk: int = 8,
) -> np.ndarray:
    """``(468,)`` the skin vertex of the head model under each MediaPipe mesh landmark.

    Args:
        videos: per head video, its fit and its ``(N, 478, 3)`` landmarks in pixels.
        max_yaw: frames turned further than this, in radians, are left out: the landmarks of a
            hidden side say nothing of the vertices there.
        cap: a landmark's distance to a vertex counts at most this, in image heights, so
            frames where a landmark is lost do not decide.
    """
    model = model.to(device)
    total = torch.zeros(NUM_MESH_LANDMARKS, ict.SKIN.stop, device=device)
    count = 0
    for fit, landmarks in videos:
        camera = fit.camera().to(device)
        usable = np.flatnonzero(
            ~np.isnan(landmarks).any(axis=(1, 2)) & (np.abs(fit.rotations[:, 2]) <= max_yaw)
        )
        for start in range(0, len(usable), chunk):
            frames = usable[start : start + chunk]

            def tensor(array: np.ndarray, frames=frames) -> torch.Tensor:
                return torch.from_numpy(np.asarray(array)[frames]).float().to(device)

            vertices = head_vertices(
                model,
                torch.from_numpy(fit.identity).float().to(device)[None],
                tensor(fit.expressions),
                tensor(fit.eye_rotations),
                tensor(fit.rotations),
                tensor(fit.translations),
                torch.from_numpy(fit.centre).float().to(device),
            )[:, ict.SKIN]
            projected = camera.project(vertices)  # (n, V, 2)
            targets = tensor(landmarks)[:, :NUM_MESH_LANDMARKS, :2]
            distance = torch.cdist(targets, projected) / camera.height  # (n, 468, V)
            total += distance.clamp_max(cap).sum(dim=0)
            count += len(frames)
    if not count:
        raise ValueError("no frontal frame with a face to find the landmarks' vertices on")
    return total.argmin(dim=1).cpu().numpy()
