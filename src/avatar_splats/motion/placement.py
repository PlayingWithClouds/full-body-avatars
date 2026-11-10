"""Where each person starts from in the fit: which way the body faces and where it stands.

People in motion videos stand, turn, bend and lie down, so fitting cannot start from one
orientation for all: started facing away, a body facing the camera fits its keypoints mirrored.
Every frame is tried with the rest pose turned to each of a set of orientations, standing and
lying; each is placed by a weak-perspective fit to the keypoints, which also gives its distance.
The orientation per frame is then the path through the frames that fits best while turning least
(dynamic programming over the orientations), so a single doubtful frame doesn't flip the body
around.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

YAWS = 8
"""Orientations tried about the vertical, evenly spaced."""
PITCHES = (0.0, np.pi / 2, -np.pi / 2)
"""Standing, lying on the back and lying on the front (turned about world x)."""
TURN_COST = 0.02
"""Cost of turning by one radian between frames, in image heights of mean keypoint error."""
MIN_KEYPOINTS = 4
"""Keypoints a frame needs to be placed; others take their neighbours' placement."""


def orientations() -> np.ndarray:
    """``(C, 3, 3)`` the orientations tried: yaw about world z after the pitch about x."""
    rotations = []
    for pitch in PITCHES:
        for k in range(YAWS):
            yaw = 2 * np.pi * k / YAWS
            rotations.append(Rotation.from_euler("xz", [pitch, yaw]).as_matrix())
    return np.array(rotations)


def weak_perspective(
    points: np.ndarray, targets: np.ndarray, weights: np.ndarray, K: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Translate camera-space point sets so their projections best match ``targets``.

    With all points at about the same depth ``d``, a pinhole projects them as ``f / d`` times
    their x, y plus an offset. Centred on their weighted mean, the scale and the offset separate
    and each has a closed form.

    Args:
        points: ``(..., K, 3)`` points in camera coordinates (x right, y down, z forward).
        targets: ``(..., K, 2)`` pixels.
        weights: ``(..., K)`` non-negative.
        K: ``(3, 3)`` intrinsics.

    Returns:
        ``(..., 3)`` translations in camera coordinates, and ``(...,)`` the weighted mean
        distances in pixels between the projected points and the targets (inf where the fit puts
        the points behind the camera or no weight is positive).
    """
    focal, principal = K[[0, 1], [0, 1]], K[:2, 2]
    total = weights.sum(axis=-1, keepdims=True)
    safe = np.where(total > 0, total, 1)
    centre = (weights[..., None] * points).sum(axis=-2) / safe
    centred = points - centre[..., None, :]
    image = (targets - principal) / focal
    offset = (weights[..., None] * image).sum(axis=-2) / safe
    spread = (weights * np.square(centred[..., :2]).sum(axis=-1)).sum(axis=-1)
    scale = (weights * (centred[..., :2] * (image - offset[..., None, :])).sum(axis=-1)).sum(
        axis=-1
    ) / np.where(spread > 0, spread, 1)
    valid = (scale > 0) & (total[..., 0] > 0)
    depth = 1 / np.where(valid, scale, 1)
    translation = np.concatenate((offset * depth[..., None], depth[..., None]), axis=-1) - centre

    moved = points + translation[..., None, :]
    projected = moved[..., :2] / np.maximum(moved[..., 2:], 1e-6) * focal + principal
    error = (weights * np.linalg.norm(projected - targets, axis=-1)).sum(axis=-1) / safe[..., 0]
    return translation, np.where(valid, error, np.inf)


def place_people(
    rest_keypoints: np.ndarray,
    keypoints: np.ndarray,
    K: np.ndarray,
    view: np.ndarray,
    min_confidence: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Orientation and translation per person and frame to start fitting from.

    Args:
        rest_keypoints: ``(C, 17, 3)`` the body's COCO keypoints in world coordinates, posed at
            the origin in each of the ``orientations()``.
        keypoints: ``(P, N, 17, 3)`` detected keypoints as x, y in pixels and confidence.
        K: ``(3, 3)`` intrinsics; ``view``: ``(4, 4)`` the camera's world-to-camera transform,
            a rotation only.

    Returns:
        ``(P, N, 3, 3)`` root rotations in world axes and ``(P, N, 3)`` world translations.
        Frames without enough keypoints take the neighbouring frames' placement.
    """
    candidates = orientations()
    people, frames = keypoints.shape[:2]
    camera = rest_keypoints @ view[:3, :3].T  # (C, 17, 3)
    turns = TURN_COST * _angles_between(candidates)
    image_height = 2 * K[1, 2]
    rotation = np.zeros((people, frames, 3, 3))
    translation = np.zeros((people, frames, 3))
    for p in range(people):
        weights = np.nan_to_num(keypoints[p, ..., 2])
        weights = weights * (weights >= min_confidence)
        placed = (weights > 0).sum(axis=1) >= MIN_KEYPOINTS
        offsets, errors = weak_perspective(
            camera[None],
            np.nan_to_num(keypoints[p, :, None, :, :2]),
            weights[:, None],
            K,
        )  # (N, C, 3), (N, C)
        cost = np.where(placed[:, None], errors / image_height, 0.0)
        path = _cheapest_path(cost, turns)
        rotation[p] = candidates[path]
        found = offsets[np.arange(frames), path]
        found[~placed | ~np.isfinite(errors[np.arange(frames), path])] = np.nan
        translation[p] = _fill(found) @ view[:3, :3]  # camera to world axes
    return rotation, translation


def _angles_between(rotations: np.ndarray) -> np.ndarray:
    """``(C, C)`` angles in radians between pairs of ``(C, 3, 3)`` rotations."""
    relative = np.einsum("aji,bjk->abik", rotations, rotations)
    cosine = (np.trace(relative, axis1=-2, axis2=-1) - 1) / 2
    return np.arccos(np.clip(cosine, -1, 1))


def _cheapest_path(cost: np.ndarray, transition: np.ndarray) -> np.ndarray:
    """``(N,)`` the choice per step minimizing the summed ``(N, C)`` costs plus ``(C, C)``
    transition costs between consecutive choices (Viterbi)."""
    frames, choices = cost.shape
    total = cost[0].copy()
    back = np.zeros((frames, choices), dtype=int)
    for n in range(1, frames):
        options = total[:, None] + transition
        back[n] = options.argmin(axis=0)
        total = options.min(axis=0) + cost[n]
    path = np.zeros(frames, dtype=int)
    path[-1] = total.argmin()
    for n in range(frames - 1, 0, -1):
        path[n - 1] = back[n, path[n]]
    return path


def _fill(values: np.ndarray) -> np.ndarray:
    """``(N, D)`` values with NaN rows interpolated from the nearest known ones."""
    known = np.flatnonzero(~np.isnan(values).any(axis=1))
    if not known.size:
        raise ValueError("no frame could be placed")
    frames = np.arange(len(values))
    return np.stack(
        [np.interp(frames, known, values[known, d]) for d in range(values.shape[1])], axis=1
    )
