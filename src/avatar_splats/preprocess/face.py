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
FACE_KEYPOINTS = slice(0, 5)
"""COCO's nose, eyes and ears."""
MIN_FACE_CONFIDENCE = 0.3
CROP_PER_FACE = 1.6
"""Half the side of a face's crop, in spans of its eye and ear keypoints."""
BLENDSHAPE_NAMES = (
    "browDownLeft",
    "browDownRight",
    "browInnerUp",
    "browOuterUpLeft",
    "browOuterUpRight",
    "cheekPuff",
    "cheekSquintLeft",
    "cheekSquintRight",
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "eyeLookDownLeft",
    "eyeLookDownRight",
    "eyeLookInLeft",
    "eyeLookInRight",
    "eyeLookOutLeft",
    "eyeLookOutRight",
    "eyeLookUpLeft",
    "eyeLookUpRight",
    "eyeSquintLeft",
    "eyeSquintRight",
    "eyeWideLeft",
    "eyeWideRight",
    "jawForward",
    "jawLeft",
    "jawOpen",
    "jawRight",
    "mouthClose",
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthFunnel",
    "mouthLeft",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthPucker",
    "mouthRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    "noseSneerLeft",
    "noseSneerRight",
)
"""MediaPipe's face blendshapes (ARKit's names, without ``tongueOut``), its ``_neutral`` left
out."""


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


def detect_blendshapes(
    frames: Sequence[np.ndarray], keypoints: np.ndarray, model_path: Path
) -> np.ndarray:
    """MediaPipe's ARKit-named blendshape weights of one person's face in RGB frames.

    In frames showing whole bodies the face is small, so it is cropped around the person's
    face keypoints first, the crop sized by the distance between the eyes and ears.

    Args:
        keypoints: ``(N, 17, 3)`` COCO body keypoints of the person.

    Returns:
        ``(N, B)`` weights in [0, 1] of ``BLENDSHAPE_NAMES``, NaN where no face was found.
    """
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,
        output_face_blendshapes=True,
    )
    index = {name: i for i, name in enumerate(BLENDSHAPE_NAMES)}
    weights = np.full((len(frames), len(BLENDSHAPE_NAMES)), np.nan)
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        for i, (frame, points) in enumerate(zip(frames, keypoints, strict=True)):
            crop = _face_crop(frame, points)
            if crop is None:
                continue
            result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=crop))
            if result.face_blendshapes:
                for category in result.face_blendshapes[0]:
                    if category.category_name in index:
                        weights[i, index[category.category_name]] = category.score
    return weights


def detect_face_blendshapes(frames: Sequence[np.ndarray], model_path: Path) -> np.ndarray:
    """MediaPipe's ARKit-named blendshape weights of the face in RGB frames of a head video,
    where the face fills the frame and needs no crop.

    Returns:
        ``(N, B)`` weights in [0, 1] of ``BLENDSHAPE_NAMES``, NaN where no face was found.
    """
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,
        output_face_blendshapes=True,
    )
    index = {name: i for i, name in enumerate(BLENDSHAPE_NAMES)}
    weights = np.full((len(frames), len(BLENDSHAPE_NAMES)), np.nan)
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        for i, frame in enumerate(frames):
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame))
            result = landmarker.detect(image)
            for category in result.face_blendshapes[0] if result.face_blendshapes else []:
                if category.category_name in index:
                    weights[i, index[category.category_name]] = category.score
    return weights


def _face_crop(frame: np.ndarray, keypoints: np.ndarray) -> np.ndarray | None:
    """The square crop around one person's face, or None when the face keypoints are unsure."""
    face = keypoints[FACE_KEYPOINTS]
    face = face[face[:, 2] >= MIN_FACE_CONFIDENCE]
    if len(face) < 3:
        return None
    centre = face[:, :2].mean(axis=0)
    half = CROP_PER_FACE * max(np.ptp(face[:, 0]), np.ptp(face[:, 1]), 8.0)
    height, width = frame.shape[:2]
    x0, y0 = np.maximum(np.floor(centre - half), 0).astype(int)
    x1, y1 = np.minimum(np.ceil(centre + half), (width, height)).astype(int)
    if x1 - x0 < 16 or y1 - y0 < 16:
        return None
    return np.ascontiguousarray(frame[y0:y1, x0:x1])
