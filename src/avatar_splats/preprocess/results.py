"""The per-frame results of preprocessing one video, and their file format."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Preprocessed:
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

    def save(self, path: str | Path) -> None:
        arrays = {name: value for name, value in vars(self).items() if value is not None}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> Preprocessed:
        with np.load(path) as data:
            return cls(**{name: data[name] for name in data.files})
