"""How far the turntable has turned in each frame, estimated from keypoints and landmarks.

These estimates initialize the fitting, which refines them. The angle is the performer's yaw: a
rotation about world +z, zero when facing the camera, positive when the nose moves to the right
of the image. Turntables need not turn at a constant speed, so every frame gets its own angle.
"""

from __future__ import annotations

import numpy as np

from avatar_splats.cameras.camera import world_to_camera

LEFT_SHOULDER, RIGHT_SHOULDER, NOSE = 5, 6, 0


def body_yaw(keypoints: np.ndarray, min_confidence: float = 0.6) -> np.ndarray:
    """Yaw per frame from COCO keypoints of a body turning on a turntable.

    The shoulders' horizontal distance in the image is the cosine of the yaw; which side of them
    the nose is on gives its sign. Frames without a confident answer are interpolated.

    Args:
        keypoints: ``(N, 17, 3)`` x, y in pixels and confidence.

    Returns:
        ``(N,)`` yaw in radians, continuous over the frames.
    """
    shoulders = keypoints[:, LEFT_SHOULDER, 0] - keypoints[:, RIGHT_SHOULDER, 0]
    cosine = np.clip(shoulders / np.nanmax(np.abs(shoulders)), -1, 1)
    nose = keypoints[:, NOSE, 0] - 0.5 * (
        keypoints[:, LEFT_SHOULDER, 0] + keypoints[:, RIGHT_SHOULDER, 0]
    )
    sine = np.sign(nose) * np.sqrt(1 - cosine**2)

    known = (keypoints[:, [LEFT_SHOULDER, RIGHT_SHOULDER, NOSE], 2] >= min_confidence).all(axis=1)
    known |= np.abs(cosine) > 0.95  # facing the camera or turned away: the sign doesn't matter
    yaw = np.where(known, np.arctan2(sine, cosine), np.nan)
    return unwrap(yaw)


def head_yaw(landmarks: np.ndarray, min_run: int = 3) -> np.ndarray:
    """Yaw per frame from face landmarks, relative to the first frame with a face.

    Each frame's landmarks are aligned rigidly to the reference frame's; the rotation's yaw is
    the head's. It underestimates turns beyond about 45 degrees, which fitting corrects. Frames
    without a face, such as the back of the head, are interpolated, and so are faces found in
    fewer than ``min_run`` consecutive frames: near profile views those are often spurious.

    Args:
        landmarks: ``(N, L, 3)`` x, y and depth in pixels; NaN where no face was found.

    Returns:
        ``(N,)`` yaw in radians, continuous over the frames.
    """
    found = _in_runs(~np.isnan(landmarks).any(axis=(1, 2)), min_run)
    reference = landmarks[np.argmax(found)]
    camera_to_world = world_to_camera()[:3, :3].T.numpy()

    yaw = np.full(len(landmarks), np.nan)
    for i in np.flatnonzero(found):
        rotation = camera_to_world @ _kabsch(reference, landmarks[i]) @ camera_to_world.T
        yaw[i] = np.arctan2(rotation[1, 0], rotation[0, 0])
    return unwrap(yaw)


def unwrap(yaw: np.ndarray) -> np.ndarray:
    """Make yaw continuous and fill NaN gaps, assuming the turntable keeps its direction.

    Between known frames the angle changes by less than half a turn, except across gaps, which
    are taken to continue in the direction the turntable turns overall.
    """
    known = np.flatnonzero(~np.isnan(yaw))
    if not known.size:
        raise ValueError("no frame has a known yaw")
    angles = yaw[known].copy()
    steps = np.angle(np.exp(1j * np.diff(angles)))
    gaps = np.diff(known) > 1
    direction = np.sign(steps[~gaps].sum()) or 1.0
    for i in np.flatnonzero(gaps):
        if np.sign(steps[i]) != direction:
            steps[i] += direction * 2 * np.pi
    angles = angles[0] + np.concatenate(([0.0], np.cumsum(steps)))
    return np.interp(np.arange(len(yaw)), known, angles)


def _in_runs(flags: np.ndarray, min_run: int) -> np.ndarray:
    """``flags`` with the True runs shorter than ``min_run`` cleared."""
    kept = flags.copy()
    edges = np.flatnonzero(np.diff(np.concatenate(([0], flags.astype(int), [0]))))
    for start, end in zip(edges[::2], edges[1::2], strict=True):
        if end - start < min_run:
            kept[start:end] = False
    return kept


def _kabsch(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """The ``(3, 3)`` rotation that best maps centred ``source`` points onto ``target`` points."""
    a = source - source.mean(axis=0)
    b = target - target.mean(axis=0)
    u, _, vt = np.linalg.svd(a.T @ b)
    flip = np.sign(np.linalg.det(vt.T @ u.T))
    return vt.T @ np.diag([1, 1, flip]) @ u.T
