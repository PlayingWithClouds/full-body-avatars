"""The eyes, for the runtime's eye material: their colours and size, measured from the head video.

The eyeballs are not drawn from the baked texture: averaged over frames that look in different
directions and blink, it ghosts. The runtime draws them instead (an iris of the performer's
colour and size, a pupil, the sclera and a glossy cornea) from what is measured here, in the
head turntable's frames that face the camera with the eyes open, and from the eyeballs' shape.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from avatar_splats.body_models.hybrid import EYES, Avatar

NOSE_TIP = 1
EYE_LANDMARKS = (
    # iris centre, iris ring, eye corners, upper and lower lid: MediaPipe's indices
    (468, (469, 470, 471, 472), (33, 133), (159, 145)),
    (473, (474, 475, 476, 477), (362, 263), (386, 374)),
)
ICT_EYE_CORNERS = ((36, 39), (42, 45))
"""The Multi-PIE landmarks at the corners of each eye."""

FRONTAL = 0.05
"""Frames count as facing the camera where the nose tip is this close to the middle between the
outer eye corners, in shares of their distance."""
OPEN = 0.7
"""Eyes count as open where the lids are this far apart, in shares of the iris' diameter."""
IRIS_RING = (0.5, 0.9)
"""Where the iris colour is sampled, in shares of its radius: outside the pupil, inside the
dark limbal ring."""
UNDER_LID = 0.4
"""The iris' top this deep, in shares of its radius, is left out: the upper lid shades it."""
SCLERA_SPAN = (1.15, 0.7)
"""Where the sclera colour is sampled beside the iris: from this many iris radii out from its
centre to this share of the way to each eye corner, where the lids close in."""
SCLERA_HEIGHT = 0.2
"""How far above and below the iris' centre the sclera is sampled, in iris radii."""


@dataclass(frozen=True)
class EyeLook:
    iris_colour: np.ndarray
    """``(3,)`` median RGB of the iris in [0, 1], as the video shows it."""
    sclera_colour: np.ndarray
    """``(3,)`` median RGB of the white of the eye beside the iris in [0, 1]."""
    iris_width: float
    """The iris' diameter in shares of the eye's width from corner to corner."""


def measure_eyes(frames: np.ndarray, landmarks: np.ndarray) -> EyeLook:
    """Measure the eyes in the frames that face the camera with open eyes.

    Args:
        frames: ``(N, H, W, 3)`` uint8 RGB.
        landmarks: ``(N, 478, 3)`` MediaPipe face landmarks, x, y in pixels; NaN where none.
    """
    iris, sclera, widths = [], [], []
    for frame, points in zip(frames, landmarks, strict=True):
        if np.isnan(points).any() or not frontal(points):
            continue
        for centre, ring, corners, lids in EYE_LANDMARKS:
            middle = points[centre, :2]
            radius = np.linalg.norm(points[list(ring), :2] - middle, axis=1).mean()
            if np.linalg.norm(points[lids[0], :2] - points[lids[1], :2]) < OPEN * 2 * radius:
                continue
            width = np.linalg.norm(points[corners[0], :2] - points[corners[1], :2])
            widths.append(2 * radius / width)
            iris.append(_iris_pixels(frame, middle, radius))
            sclera.append(_sclera_pixels(frame, middle, radius, points[list(corners), 0]))
    if not widths:
        raise ValueError("no frame faces the camera with open eyes")
    # The lids shade the white near them, and the inner corner is pink: its open, lit part is
    # the brighter half of what is sampled. The runtime shades it towards the lids itself.
    sclera = np.concatenate(sclera)
    brightness = sclera.sum(axis=1)
    lit = sclera[brightness >= np.median(brightness)]
    return EyeLook(
        iris_colour=np.median(np.concatenate(iris), axis=0) / 255,
        sclera_colour=np.median(lit, axis=0) / 255,
        iris_width=float(np.median(widths)),
    )


def eye_metadata(avatar: Avatar, look: EyeLook, ict_landmarks: list[int]) -> dict:
    """The eyes' entry in the avatar file's metadata.

    The iris spans ``iris_angle`` radians from each eyeball's front, measured at its centre: the
    iris' measured share of the eye's width, times the width of the avatar's eyes, on an
    eyeball of the fitted radius. Each eyeball lists its centre and radius in the rest pose and
    two of its vertices, its front (``apex``) and its top, which show how it is turned when
    posed.
    """
    landmarks = avatar.head_start + np.asarray(ict_landmarks)
    corners = avatar.rest_vertices[landmarks[np.asarray(ICT_EYE_CORNERS)]]
    eye_width = np.linalg.norm(corners[:, 0] - corners[:, 1], axis=-1).mean()

    eye_vertices = np.unique(avatar.faces[avatar.face_parts == EYES])
    eyeballs = []
    for side in (eye_vertices[avatar.rest_vertices[eye_vertices, 0] > 0],
                 eye_vertices[avatar.rest_vertices[eye_vertices, 0] < 0]):  # fmt: skip
        points = avatar.rest_vertices[side]
        centre, radius = _sphere(points)
        eyeballs.append(
            {
                "side": "left" if centre[0] > 0 else "right",  # the body's left is +x
                "centre": centre.tolist(),
                "radius": radius,
                "apex": int(side[np.argmin(points[:, 1])]),  # the face looks along -y
                "top": int(side[np.argmax(points[:, 2])]),
            }
        )
    radius = np.mean([eyeball["radius"] for eyeball in eyeballs])
    return {
        "iris_colour": look.iris_colour.tolist(),
        "sclera_colour": look.sclera_colour.tolist(),
        "iris_angle": float(np.arcsin(min(1.0, look.iris_width * eye_width / 2 / radius))),
        "eyeballs": eyeballs,
    }


def frontal(points: np.ndarray) -> bool:
    """Whether ``(478, 3)`` MediaPipe landmarks show a face turned to the camera."""
    outer = points[[EYE_LANDMARKS[0][2][0], EYE_LANDMARKS[1][2][1]], 0]
    return abs(points[NOSE_TIP, 0] - outer.mean()) < FRONTAL * abs(outer[1] - outer[0])


def _disc(frame: np.ndarray, middle: np.ndarray, extent: float) -> tuple[np.ndarray, ...]:
    """Pixel coordinates within ``extent`` of ``middle`` and their colours."""
    low = np.maximum(np.floor(middle - extent).astype(int), 0)
    high = np.minimum(np.ceil(middle + extent).astype(int) + 1, frame.shape[1::-1])
    ys, xs = np.mgrid[low[1] : high[1], low[0] : high[0]]
    return xs - middle[0], ys - middle[1], frame[ys, xs].reshape(*ys.shape, 3)


def _iris_pixels(frame: np.ndarray, middle: np.ndarray, radius: float) -> np.ndarray:
    dx, dy, colours = _disc(frame, middle, radius)
    distance = np.hypot(dx, dy) / radius
    inside = (distance > IRIS_RING[0]) & (distance < IRIS_RING[1]) & (dy > -UNDER_LID * radius)
    return colours[inside].astype(np.float64)


def _sclera_pixels(
    frame: np.ndarray, middle: np.ndarray, radius: float, corners_x: np.ndarray
) -> np.ndarray:
    inner, share = SCLERA_SPAN
    dx, dy, colours = _disc(frame, middle, np.abs(corners_x - middle[0]).max())
    taken = np.zeros(dx.shape, bool)
    for corner in corners_x - middle[0]:
        along = dx * np.sign(corner)
        taken |= (along > inner * radius) & (along < share * abs(corner))
    taken &= np.abs(dy) < SCLERA_HEIGHT * radius
    return colours[taken].astype(np.float64)


def _sphere(points: np.ndarray) -> tuple[np.ndarray, float]:
    """The least-squares sphere through ``(P, 3)`` points: its centre and radius."""
    system = np.c_[2 * points, np.ones(len(points))]
    solution = np.linalg.lstsq(system, (points**2).sum(axis=1), rcond=None)[0]
    centre = solution[:3]
    return centre, float(np.sqrt(solution[3] + centre @ centre))
