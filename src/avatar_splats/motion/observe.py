"""What a motion video shows of each person in it: masks, body keypoints, hands and face.

The people are found on the first stretch of the video, on the frame where most of them are
detected, and SAM2 follows each from there. Long videos are read and tracked a stretch of
``track_frames`` at a time, each stretch prompted with the masks the one before ended on, so a
person keeps their identity through the whole video while only one stretch is held in memory.
Each person's keypoints, hands and face are estimated within their own mask's box, with the same
models the capture's preprocessing uses.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from avatar_splats.cameras import Camera
from avatar_splats.config import MotionConfig, PreprocessConfig
from avatar_splats.data import probe, read_frames
from avatar_splats.preprocess.detection import detect_people
from avatar_splats.preprocess.face import (
    BLENDSHAPE_NAMES,
    detect_blendshapes,
    download_face_landmarker,
)
from avatar_splats.preprocess.hands import detect_hands, download_hand_landmarker
from avatar_splats.preprocess.keypoints import estimate_keypoints
from avatar_splats.preprocess.segmentation import mask_boxes, track_people
from avatar_splats.records import ArrayRecord

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Observations(ArrayRecord):
    """Per person and frame, everything the models found in a motion video.

    People are in the order they were first detected, the most confident first. Where a person
    was not found, the arrays hold NaN (and the masks are empty).
    """

    frame_indices: np.ndarray
    """``(N,)`` int: index of each processed frame in the video."""
    fps: float
    """Frames per second of the processed frames."""
    image_size: np.ndarray
    """``(2,)`` width and height of the video."""
    mask_size: np.ndarray
    """``(2,)`` width and height of the masks: the size the bodies are fitted at."""
    masks: np.ndarray
    """``(P, N, h, ceil(w / 8))`` uint8: each person's silhouette at ``mask_size``, eight
    pixels per byte along each row (``np.packbits``); see ``mask_frames``."""
    boxes: np.ndarray
    """``(P, N, 4)`` the silhouettes' bounding boxes as x0, y0, x1, y1 in video pixels."""
    keypoints: np.ndarray
    """``(P, N, 17, 3)`` COCO body keypoints as x, y in video pixels and confidence."""
    hands: np.ndarray
    """``(P, N, 2, 21, 3)`` MediaPipe landmarks of the left and right hand as x, y in video
    pixels and confidence."""
    blendshapes: np.ndarray
    """``(P, N, B)`` MediaPipe's ARKit-named blendshape weights of the face, in [0, 1]."""
    blendshape_names: np.ndarray
    """``(B,)`` str: MediaPipe's names of the blendshapes (``eyeBlinkLeft``, ...)."""
    ignored: np.ndarray | None = None
    """Packed like ``masks``: pixels of each person where silhouettes are not compared, such as
    hair, which the bald body has no surface for; see ``ignored_frames``."""
    extra_keypoints: np.ndarray | None = None
    """``(P, N, 12, 3)`` the ``preprocess.keypoints.EXTRA_KEYPOINTS`` (toes, heels, elbows' back
    and front, shoulders' tops) as x, y in video pixels and confidence; NaN where unknown."""
    keypoint_depths: np.ndarray | None = None
    """``(P, N, 17)`` the estimated depth of the skin at each confident COCO keypoint's pixel
    (``preprocess.surface.lift_keypoints``), its own or what hides it, in metres up to scale;
    NaN where unknown. Fitting starts from the 3D points they give and keeps
    the keypoints behind that skin and near it."""
    normals: np.ndarray | None = None
    """``(P, N, h, w, 3)`` float16 each person's estimated surface normals at ``mask_size``,
    in the camera's axes (x right, y down, z forward); zero where unknown
    (``preprocess.surface``). The fit turns the body's to them."""
    depths: np.ndarray | None = None
    """``(P, N, D, h, w)`` float16 D estimates of each person's depth at ``mask_size``, in
    metres up to scale; NaN where unknown. The fit matches the body's relief to them."""

    @property
    def num_people(self) -> int:
        return len(self.masks)

    def mask_frames(self, frames: slice | list[int]) -> np.ndarray:
        """``(P, n, h, w)`` bool masks of the given frames."""
        width = int(self.mask_size[0])
        return np.unpackbits(self.masks[:, frames], axis=-1, count=width).astype(bool)

    def ignored_frames(self, frames: slice | list[int]) -> np.ndarray:
        """``(P, n, h, w)`` bool ignored pixels of the given frames; none without ``ignored``."""
        if self.ignored is None:
            return np.zeros_like(self.mask_frames(frames))
        width = int(self.mask_size[0])
        return np.unpackbits(self.ignored[:, frames], axis=-1, count=width).astype(bool)


def mask_size(image_size: tuple[int, int], resolution: int) -> tuple[int, int]:
    """Width and height of masks whose long side is ``resolution``, as ``Camera.scaled``
    rounds them."""
    width, height = image_size
    camera = Camera.fixed(width, height, 30.0).scaled(resolution / max(width, height))
    return camera.width, camera.height


def observe_people(path: Path, motion: MotionConfig, models: PreprocessConfig) -> Observations:
    """Find up to ``motion.max_people`` people in a video and observe each through it."""
    info = probe(path)
    size = mask_size((info.width, info.height), motion.resolution)
    hand_model = download_hand_landmarker(models.hand_landmarker)
    face_model = download_face_landmarker(models.face_landmarker)
    frames = read_frames(path, step=motion.frame_step)

    stretches: list[dict[str, np.ndarray]] = []
    last_frame = last_masks = last_boxes = None
    while stretch := list(itertools.islice(frames, motion.track_frames)):
        if last_frame is None:
            prompt, boxes = _prompt_frame(stretch, motion, models)
            logger.info("%d people found on frame %d", len(boxes), prompt)
            masks = track_people(stretch, prompt, models.segmenter, models.device, boxes=boxes)
        else:
            # Go on from the last frame tracked, which is tracked again and then dropped.
            masks = track_people(
                [last_frame, *stretch],
                0,
                models.segmenter,
                models.device,
                boxes=last_boxes,
                masks=last_masks,
            )[:, 1:]
        last_frame, last_masks = stretch[-1], masks[:, -1]
        observed = _observe_stretch(stretch, masks, size, hand_model, face_model, models)
        boxes = observed["boxes"][:, -1]
        known = ~np.isnan(boxes).any(axis=1)
        last_boxes = boxes if last_boxes is None else np.where(known[:, None], boxes, last_boxes)
        stretches.append(observed)
        done = sum(len(s["boxes"][0]) for s in stretches)
        logger.info(
            "observed %d frames, %.0f s of video", done, done / info.fps * motion.frame_step
        )

    joined = {key: np.concatenate([s[key] for s in stretches], axis=1) for key in stretches[0]}
    frame_count = joined["boxes"].shape[1]
    return Observations(
        frame_indices=np.arange(frame_count) * motion.frame_step,
        fps=info.fps / motion.frame_step,
        image_size=np.array([info.width, info.height]),
        mask_size=np.array(size),
        masks=joined["masks"],
        boxes=joined["boxes"],
        keypoints=joined["keypoints"],
        hands=joined["hands"],
        blendshapes=joined["blendshapes"],
        blendshape_names=np.array(BLENDSHAPE_NAMES),
    )


def _observe_stretch(
    frames: list[np.ndarray],
    masks: np.ndarray,
    size: tuple[int, int],
    hand_model: Path,
    face_model: Path,
    models: PreprocessConfig,
) -> dict:
    """Keypoints, hands and face of each person in a stretch of frames, and their masks at
    ``size``, packed."""
    people = []
    for person_masks in masks:
        boxes = mask_boxes(person_masks)
        keypoints = estimate_keypoints(frames, boxes, models.pose_estimator, models.device)
        hands = detect_hands(frames, keypoints, hand_model)
        people.append((boxes, keypoints, hands, detect_blendshapes(frames, keypoints, face_model)))
    small = F.interpolate(
        torch.from_numpy(masks).float().flatten(0, 1)[:, None], size[::-1], mode="area"
    )[:, 0].unflatten(0, masks.shape[:2])
    return {
        "masks": np.packbits((small > 0.5).numpy(), axis=-1),
        "boxes": np.stack([p[0] for p in people]),
        "keypoints": np.stack([p[1] for p in people]),
        "hands": np.stack([p[2] for p in people]),
        "blendshapes": np.stack([p[3] for p in people]),
    }


def _prompt_frame(
    frames: list[np.ndarray], motion: MotionConfig, models: PreprocessConfig
) -> tuple[int, np.ndarray]:
    """The frame where the most people (up to ``max_people``) are detected, the surest of those
    frames, and the ``(P, 5)`` boxes and confidences found on it.

    Raises:
        LookupError: when no frame shows a person.
    """
    searched = list(range(0, len(frames), motion.detection_every))
    found = detect_people(
        [frames[i] for i in searched], models.detector, models.device, motion.detection_threshold
    )
    found = [boxes[: motion.max_people] for boxes in found]
    best = max(range(len(found)), key=lambda i: (len(found[i]), found[i][:, 4].sum()))
    if not len(found[best]):
        raise LookupError("no person detected")
    return searched[best], found[best]
