"""Face landmarks with MediaPipe's face landmarker."""

from __future__ import annotations

import urllib.request
from collections.abc import Sequence
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
NUM_FACE_LANDMARKS = 478


def download_face_landmarker(path: Path) -> Path:
    """Download the face landmarker model to ``path`` unless it is already there."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".part")
        urllib.request.urlretrieve(FACE_LANDMARKER_URL, partial)
        partial.rename(path)
    return path


def detect_face_landmarks(frames: Sequence[np.ndarray], model_path: Path, fps: float) -> np.ndarray:
    """Track the face landmarks through consecutive RGB frames sampled at ``fps``.

    Returns:
        ``(N, 478, 3)`` landmarks as x, y in pixels and depth in pixels of image width,
        relative to the head's center; NaN where no face was found.
    """
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
    )
    landmarks = np.full((len(frames), NUM_FACE_LANDMARKS, 3), np.nan)
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        for i, frame in enumerate(frames):
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame))
            result = landmarker.detect_for_video(image, timestamp_ms=round(1000 * i / fps))
            if result.face_landmarks:
                height, width = frame.shape[:2]
                points = [(p.x, p.y, p.z) for p in result.face_landmarks[0]]
                landmarks[i] = np.array(points) * (width, height, width)
    return landmarks
