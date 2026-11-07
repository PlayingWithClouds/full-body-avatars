"""Hand keypoints with MediaPipe's hand landmarker, run on crops around the body's wrists.

In a full-body frame the hands are small, so each is cropped around its wrist keypoint, with the
crop sized by the forearm, and the landmarker looks at the crop alone. Of the hands it finds
there, the one whose wrist is nearest the body's wrist keypoint is kept, so left and right
follow the body keypoints rather than the landmarker's guess of handedness.
"""

from __future__ import annotations

import urllib.request
from collections.abc import Sequence
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
NUM_HAND_LANDMARKS = 21
"""MediaPipe's hand landmarks: the wrist, then four per finger from the thumb to the little
finger, from the knuckle nearest the wrist to the tip."""
COCO_ELBOWS, COCO_WRISTS = (7, 8), (9, 10)
"""The body keypoints of the left and right elbow and wrist."""
MIN_WRIST_CONFIDENCE = 0.3
CROP_PER_FOREARM = 1.2
"""Half the side of a hand's crop, in forearm lengths."""


def download_hand_landmarker(path: Path) -> Path:
    """Download the hand landmarker model to ``path`` unless it is already there."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".part")
        urllib.request.urlretrieve(HAND_LANDMARKER_URL, partial)
        partial.rename(path)
    return path


def detect_hands(
    frames: Sequence[np.ndarray], keypoints: np.ndarray, model_path: Path
) -> np.ndarray:
    """Find the left and right hand's landmarks in RGB frames.

    Args:
        keypoints: ``(N, 17, 3)`` COCO body keypoints of the frames.

    Returns:
        ``(N, 2, 21, 3)`` landmarks of the left and right hand as x, y in pixels and the
        landmarker's confidence in the hand; NaN where it was not found.
    """
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
    )
    hands = np.full((len(frames), 2, NUM_HAND_LANDMARKS, 3), np.nan)
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        for i, frame in enumerate(frames):
            for side in range(2):
                elbow, wrist = keypoints[i, COCO_ELBOWS[side]], keypoints[i, COCO_WRISTS[side]]
                if not min(elbow[2], wrist[2]) >= MIN_WRIST_CONFIDENCE:
                    continue
                half = CROP_PER_FOREARM * np.linalg.norm(wrist[:2] - elbow[:2])
                found = _landmarks_in_crop(landmarker, frame, wrist[:2], half)
                if found:
                    hands[i, side] = min(
                        found, key=lambda hand: np.linalg.norm(hand[0, :2] - wrist[:2])
                    )
    return hands


def _landmarks_in_crop(
    landmarker: vision.HandLandmarker, frame: np.ndarray, centre: np.ndarray, half: float
) -> list[np.ndarray]:
    """``(21, 3)`` landmarks in frame pixels and confidence of every hand found in the square
    crop of half side ``half`` around ``centre``."""
    height, width = frame.shape[:2]
    x0, y0 = np.maximum(np.floor(centre - half), 0).astype(int)
    x1, y1 = np.minimum(np.ceil(centre + half), (width, height)).astype(int)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return []
    crop = np.ascontiguousarray(frame[y0:y1, x0:x1])
    result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=crop))
    found = []
    for landmarks, handedness in zip(result.hand_landmarks, result.handedness, strict=True):
        points = np.array([(p.x * (x1 - x0) + x0, p.y * (y1 - y0) + y0) for p in landmarks])
        confidence = np.full((len(points), 1), handedness[0].score)
        found.append(np.concatenate((points, confidence), axis=1))
    return found
