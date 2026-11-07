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
    face_landmarks: np.ndarray | None = None
    """``(N, 478, 3)`` float: MediaPipe face landmarks as x, y in pixels and depth, in pixels
    of image width, relative to the head's center."""
