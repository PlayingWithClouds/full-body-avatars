"""The per-frame results of preprocessing one video."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from avatar_splats.records import ArrayRecord


@dataclass(frozen=True)
class Preprocessed(ArrayRecord):
    """Everything preprocessing found in one video.

    Every array has one entry per processed frame. Frames where nothing was found hold NaN.
    """

    frame_indices: np.ndarray
    """``(N,)`` int: index of each processed frame in the video."""
    masks: np.ndarray
    """``(N, H, W)`` bool: the performer's silhouette."""
    boxes: np.ndarray
    """``(N, 4)`` float: the silhouette's bounding box as x0, y0, x1, y1 in pixels."""
    keypoints: np.ndarray | None = None
    """``(N, 17, 3)`` float: COCO body keypoints as x, y in pixels and confidence."""
    hands: np.ndarray | None = None
    """``(N, 2, 21, 3)`` float: MediaPipe landmarks of the left and right hand as x, y in pixels
    and confidence."""
    face_landmarks: np.ndarray | None = None
    """``(N, 478, 3)`` float: MediaPipe face landmarks as x, y in pixels and depth, in pixels
    of image width, relative to the head's center."""
    ignored: np.ndarray | None = None
    """``(N, H, W)`` bool: pixels where the fits do not compare silhouettes, such as hair, which
    the bald avatar has no surface for."""
