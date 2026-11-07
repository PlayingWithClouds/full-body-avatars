"""Preprocessing the videos of a capture."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np

from avatar_splats.config import VIDEOS, Config, PreprocessConfig
from avatar_splats.data import probe, read_frames
from avatar_splats.preprocess.detection import detect_person
from avatar_splats.preprocess.face import detect_face_landmarks, download_face_landmarker
from avatar_splats.preprocess.hands import detect_hands, download_hand_landmarker
from avatar_splats.preprocess.keypoints import estimate_keypoints
from avatar_splats.preprocess.results import Preprocessed
from avatar_splats.preprocess.segmentation import mask_boxes, track_person


def preprocess_video(
    path: Path, config: PreprocessConfig, *, keypoints: bool, face: bool
) -> Preprocessed:
    """Segment the performer in every ``frame_step``-th frame and run the requested models."""
    frames = list(read_frames(path, step=config.frame_step))
    box = detect_person(frames[0], config.detector, config.device)
    masks = track_person(frames, box, config.segmenter, config.device)
    boxes = mask_boxes(masks)

    body_keypoints = hands = face_landmarks = None
    if keypoints:
        body_keypoints = estimate_keypoints(frames, boxes, config.pose_estimator, config.device)
        hands = detect_hands(
            frames, body_keypoints, download_hand_landmarker(config.hand_landmarker)
        )
    if face:
        model = download_face_landmarker(config.face_landmarker)
        fps = probe(path).fps / config.frame_step
        face_landmarks = detect_face_landmarks(frames, model, fps)

    return Preprocessed(
        frame_indices=np.arange(len(frames)) * config.frame_step,
        masks=masks,
        boxes=boxes,
        keypoints=body_keypoints,
        hands=hands,
        face_landmarks=face_landmarks,
    )


def preprocess_capture(config: Config, videos: Iterable[str] = VIDEOS) -> dict[str, Path]:
    """Preprocess the named videos of the capture and return where each result was written.

    The body video gets body and hand keypoints, the head videos face landmarks.
    """
    written = {}
    for name in videos:
        result = preprocess_video(
            getattr(config.capture, name),
            config.preprocess,
            keypoints=name == "body",
            face=name != "body",
        )
        written[name] = preprocessed_path(config, name)
        result.save(written[name])
    return written


def preprocessed_path(config: Config, video: str) -> Path:
    return config.outputs / "preprocess" / f"{video}.npz"
