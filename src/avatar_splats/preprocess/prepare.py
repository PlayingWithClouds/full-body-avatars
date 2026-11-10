"""Preparing a stretch of an ordinary video: one person, followed through it, with hair, in any
pose.

The person is picked on one frame (the largest one there) and followed by the detector: in each
frame, the box overlapping theirs in the frame before the most. The pose and face models see
that box alone: everything outside it (widened by ``isolate_margin``) is greyed out first. Who
else is in the box stays: interlocked people would lose parts of her to any cut through it, so
``overlap`` says how much of her box other people's boxes cover, and later steps score such
frames down. A frame goes on where ViTPose places at least ``min_placed`` of the 17 body
keypoints, ``min_face_placed`` of them on the face, so close-ups of the head or upper body go on
too, but not ones without the face; a close-up of the face alone goes on only where MediaPipe
finds the whole face in it.

Her outline and body parts (Sapiens2, ``BodyParts``) are not made here: they are the costliest
model, and only the frames chosen to fit need them. Per frame:

- COCO's 17 body keypoints from ViTPose, in the person's box (Sapiens2 pose and YOLO-pose put
  them off more often on this footage),
- 21 landmarks per hand from MediaPipe, in crops around the wrists, on the GPU,
- 478 face landmarks from MediaPipe, in a crop around the face keypoints, on the GPU,
- face parts (brows, eyes, irises, nose, lips, mouth opening) filled from the face landmarks,
  whose outlines are exact even where the face is a few dozen pixels.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, fields
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn.functional as F
from mediapipe.tasks.python import vision

from avatar_splats.preprocess.detection import detect_people, load_detector
from avatar_splats.preprocess.gpu_tasks import create_task
from avatar_splats.preprocess.hands import detect_hands
from avatar_splats.preprocess.hands import hand_landmarker as open_hand_landmarker
from avatar_splats.preprocess.keypoints import estimate_keypoints, load_vitpose
from avatar_splats.preprocess.triage import focus
from avatar_splats.records import ArrayRecord

PART_NAMES = (
    "Background", "Apparel", "Eyeglass", "Face_Neck", "Hair", "Left_Foot", "Left_Hand",
    "Left_Lower_Arm", "Left_Lower_Leg", "Left_Shoe", "Left_Sock", "Left_Upper_Arm",
    "Left_Upper_Leg", "Lower_Clothing", "Right_Foot", "Right_Hand", "Right_Lower_Arm",
    "Right_Lower_Leg", "Right_Shoe", "Right_Sock", "Right_Upper_Arm", "Right_Upper_Leg", "Torso",
    "Upper_Clothing", "Lower_Lip", "Upper_Lip", "Lower_Teeth", "Upper_Teeth", "Tongue",
)  # fmt: skip
"""Sapiens2's body part labels, in its order."""
NOT_THEM = 255
"""The body part label outside the person's outline."""
FACE_PART_NAMES = ("none", "nose", "brows", "eyes", "irises", "lips", "mouth opening")
GREY = np.array([124, 116, 104], np.uint8)
"""The models' mean colour: reads as nothing."""
INPUT_HEIGHT, INPUT_WIDTH = 1024, 768
"""Sapiens2's input; it stretches whatever it gets to this size."""
ZOOM_JUMP = 1.5
"""Factor by which the face's size in the image changes from one kept frame to the next at a
cut the shot detection missed (one to a close-up on the same backdrop)."""


@dataclass(frozen=True)
class PrepareSettings:
    min_detection: float
    """The detector's score to accept a person."""
    min_iou: float
    """A box overlapping the person's in the frame before by this much (IoU) is theirs."""
    isolate_margin: float
    """The box is widened by this share of its larger side before the rest is greyed."""
    pose_model: str
    """A ViTPose model (``preprocess.keypoints``)."""
    keypoint_seen: float
    """A keypoint's confidence to count as placed."""
    min_placed: int
    """Keypoints a frame needs placed to go on: few keeps close-ups of the head or upper body,
    which show the most detail."""
    min_face_placed: int
    """Of those, face keypoints (nose, eyes, ears) a frame needs placed: without the face a
    close-up cannot be placed on the body. Where these are all that is placed, MediaPipe has to
    find the whole face too."""
    pose_batch: int
    min_keypoint: float
    """Face keypoints below this confidence do not place the face crop."""
    landmark_crop: float
    """The face crop for the landmarks, in spans of the face keypoints."""


@dataclass(frozen=True)
class PreparedShot(ArrayRecord):
    """One person through a stretch of a video, at the frames' prepared size.

    Pixel coordinates are in the frames scaled by ``scale``. The box and keypoints are there
    for every frame the person was found in, rejected ones too (to see why they were); the hands
    and face only for ``usable`` frames. Where the person was not found the points are NaN.
    """

    frame_indices: np.ndarray
    """``(N,)`` int: the video frames."""
    scale: float
    """The prepared frames' size over the video's."""
    shot: int
    """The shot's rank in the triage, best first."""
    shot_score: float
    image_size: np.ndarray
    """``(2,)`` width and height of the prepared frames."""
    detection: np.ndarray
    """``(N,)`` the detector's score of their box; 0 where they were not found."""
    overlap: np.ndarray
    """``(N,)`` the largest share of their box another detected person's box covers: near 1
    where someone is interlocked with them, whose body their box then holds too."""
    usable: np.ndarray
    """``(N,)`` bool: frames that go on: found, at least ``min_placed`` keypoints placed and
    ``min_face_placed`` of them on the face."""
    sharpness: np.ndarray
    """``(N,)`` the ``triage.focus`` of their box; NaN where they were not found."""
    boxes: np.ndarray
    """``(N, 4)`` the detector's x0, y0, x1, y1; NaN where they were not found."""
    keypoints: np.ndarray
    """``(N, 17, 3)`` COCO keypoints from ViTPose, x, y and confidence."""
    hands: np.ndarray
    """``(N, 2, 21, 3)`` MediaPipe's left and right hand, x, y and confidence."""
    face_landmarks: np.ndarray
    """``(N, 478, 3)`` MediaPipe's, x, y and depth in pixels."""
    face_confidence: np.ndarray
    """``(N,)`` the mean confidence of the face keypoints that placed the face crop."""
    face_parts: np.ndarray
    """``(N, H, W)`` uint8 index into ``FACE_PART_NAMES``."""


def is_prepared(path: Path, first: int, frame_step: int, count: int) -> bool:
    """Whether ``path`` holds a ``PreparedShot`` of video frames ``first``, ``first +
    frame_step``, ... (at most ``count``, fewer at the video's end), with exactly this version's
    fields: a piece of an earlier run may be another stretch, or lack or hold other fields."""
    if not path.exists():
        return False
    with np.load(path) as data:
        if set(data.files) != {field.name for field in fields(PreparedShot)}:
            return False
        frames = data["frame_indices"]
    return (
        0 < len(frames) <= count
        and int(frames[0]) == first
        and bool((np.diff(frames) == frame_step).all())
    )


class Preparer:
    """The models, loaded once for every stretch prepared."""

    def __init__(
        self,
        settings: PrepareSettings,
        detector: str,
        hand_landmarker: Path,
        face_landmarker: Path,
        device: str,
    ):
        self.settings, self.detector, self.device = settings, detector, device
        self.hand_landmarker, self.face_landmarker = hand_landmarker, face_landmarker
        # loaded and compiled once: per stretch, loading and compiling the detector took 8 s
        self.detector_models = load_detector(detector, device, compiled=True)
        self.pose_models = load_vitpose(settings.pose_model, device, compiled=True)
        self.hands = open_hand_landmarker(hand_landmarker, gpu=True)
        # On the GPU where MediaPipe gets one (``gpu_tasks``): about twice as fast per face.
        self.landmarker = create_task(
            vision.FaceLandmarker,
            vision.FaceLandmarkerOptions,
            face_landmarker,
            True,
            running_mode=vision.RunningMode.IMAGE,
        )

    def prepare(
        self,
        frames: list[np.ndarray],
        frame_indices: np.ndarray,
        scale: float,
        select: int,
        shot: int = 0,
        shot_score: float = float("nan"),
        timings: dict[str, float] | None = None,
    ) -> PreparedShot | None:
        """Follow the largest person on frame ``select`` through ``frames`` and run every model
        on them. None when nobody is found there."""
        timings = {} if timings is None else timings
        s, device = self.settings, self.device
        count = len(frames)
        height, width = frames[0].shape[:2]

        with _timed(timings, "detect people"):
            found = detect_people(
                frames,
                self.detector,
                device,
                s.min_detection,
                s.pose_batch,
                compiled=True,
                models=self.detector_models,
            )
        if not len(found[select]):
            return None
        followed = follow(found, select, s.min_iou)
        detected = np.isfinite(followed[:, 0])
        boxes = followed[:, :4]
        overlap = np.array([covered_share(boxes[i], found[i]) for i in range(count)])

        with _timed(timings, "isolate the person"):
            alone = [boxed(frames[i], boxes[i], s.isolate_margin) for i in range(count)]

        def isolated(i):
            return alone[i]

        with _timed(timings, "keypoints (ViTPose)"):
            keypoints = estimate_keypoints(
                alone,
                boxes,
                s.pose_model,
                device,
                s.pose_batch,
                compiled=True,
                models=self.pose_models,
            )
        # Rejected: not found, too few keypoints placed, or too few on the face.
        sure = np.nan_to_num(keypoints[..., 2]) >= s.keypoint_seen
        usable = detected & (sure.sum(-1) >= s.min_placed)
        usable &= sure[:, :5].sum(-1) >= s.min_face_placed
        sharpness = np.array([focus(frames[i], boxes[i]) for i in range(count)])
        with _timed(timings, "face landmarks"):
            accepted = np.where(usable[:, None, None], keypoints, np.nan)
            face_landmarks, face_confidence = self._faces(isolated, accepted, width, height)
        # Where only the face's keypoints are placed, the frame is a close-up of the face: it
        # goes on only if MediaPipe finds the whole face in it.
        face_only = ~sure[:, 5:].any(axis=-1)
        found_face = np.isfinite(face_landmarks).all(axis=(1, 2))
        usable &= ~face_only | found_face
        face_landmarks[~usable] = np.nan
        face_confidence[~usable] = np.nan
        # The rest only where the frame goes on.
        accepted = np.where(usable[:, None, None], keypoints, np.nan)
        with _timed(timings, "hands"):
            hands = detect_hands(alone, accepted, gpu=True, landmarker=self.hands)
        with _timed(timings, "face parts (landmarks)"):
            face_parts = np.zeros((count, height, width), np.uint8)
            for i in range(count):
                if np.isfinite(face_landmarks[i]).all():
                    face_parts[i] = face_parts_of(face_landmarks[i], (height, width))
        return PreparedShot(
            frame_indices=np.asarray(frame_indices),
            scale=float(scale),
            shot=int(shot),
            shot_score=float(shot_score),
            image_size=np.array([width, height]),
            detection=np.nan_to_num(followed[:, 4]),
            overlap=overlap,
            usable=usable,
            sharpness=sharpness,
            boxes=boxes,
            keypoints=keypoints,
            hands=hands,
            face_landmarks=face_landmarks,
            face_confidence=face_confidence,
            face_parts=face_parts,
        )

    def _faces(self, isolated, keypoints, width, height):
        count = len(keypoints)
        landmarks = np.full((count, 478, 3), np.nan)
        confidence = np.full(count, np.nan)
        landmarker = self.landmarker
        for i in range(count):
            box = face_box(
                keypoints[i],
                self.settings.landmark_crop,
                self.settings.min_keypoint,
                (width, height),
            )
            if box is None:
                continue
            x0, y0, x1, y1 = box
            crop = np.ascontiguousarray(isolated(i)[y0:y1, x0:x1])
            result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=crop))
            if result.face_landmarks:
                points = np.array([(q.x, q.y, q.z) for q in result.face_landmarks[0]])
                landmarks[i] = points * (x1 - x0, y1 - y0, x1 - x0) + (x0, y0, 0)
                face = keypoints[i, :5, 2]
                confidence[i] = face[face >= self.settings.min_keypoint].mean()
        return landmarks, confidence


def boxed(frame: np.ndarray, box: np.ndarray, margin: float) -> np.ndarray:
    """``(H, W, 3)`` the frame grey outside the box, widened by ``margin`` of its larger side;
    all grey where the box is not finite."""
    out = np.empty_like(frame)
    out[:] = GREY
    if not np.isfinite(box).all():
        return out
    x0, y0, x1, y1 = box
    pad = margin * max(x1 - x0, y1 - y0)
    height, width = frame.shape[:2]
    x0, y0 = max(int(x0 - pad), 0), max(int(y0 - pad), 0)
    x1, y1 = min(int(np.ceil(x1 + pad)), width), min(int(np.ceil(y1 + pad)), height)
    out[y0:y1, x0:x1] = frame[y0:y1, x0:x1]
    return out


def covered_share(box: np.ndarray, found: np.ndarray) -> float:
    """The largest share of ``box`` (x0, y0, x1, y1) one of the other ``(K, 5)`` detected boxes
    covers; the one most like ``box`` is its own. 0 where alone, NaN where ``box`` is not
    finite."""
    if not np.isfinite(box).all():
        return float("nan")
    if len(found) < 2:
        return 0.0
    others = np.delete(found[:, :4], int(box_iou(box, found[:, :4]).argmax()), axis=0)
    width = (np.minimum(box[2], others[:, 2]) - np.maximum(box[0], others[:, 0])).clip(0)
    height = (np.minimum(box[3], others[:, 3]) - np.maximum(box[1], others[:, 1])).clip(0)
    area = max((box[2] - box[0]) * (box[3] - box[1]), 1e-6)
    return float((width * height).max() / area)


class BodyParts:
    """Sapiens2's body parts of one person per frame: their outline, hair and clothes.

    The costliest model of the preparation, so run only on the frames chosen to fit."""

    def __init__(self, model: str, batch: int, person_margin: float, device: str):
        """``person_margin``: the crop's margin around a person's box, in shares of its larger
        side."""
        from transformers import AutoImageProcessor, Sapiens2ForSemanticSegmentation

        self.batch, self.person_margin, self.device = batch, person_margin, device
        self.processor = AutoImageProcessor.from_pretrained(model)
        self.model = torch.compile(
            Sapiens2ForSemanticSegmentation.from_pretrained(model, dtype=torch.bfloat16)
            .to(device)
            .eval()
        )

    def __call__(self, frames, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(N, H, W)`` uint8 labels (``PART_NAMES``, ``NOT_THEM`` outside the person) and
        ``(N,)`` certainty of ``(N, H, W, 3)`` frames' person in ``(N, 4)`` boxes; see
        ``_parts``."""
        valid = np.flatnonzero(np.isfinite(boxes).all(axis=1))
        return self._parts(frames, boxes, valid)

    @torch.inference_mode()
    def _parts(self, frames, boxes, valid):
        """``(N, H, W)`` Sapiens2's labels in a crop around each box, kept where the labelled
        regions lie mostly inside the box (widened by half ``person_margin``): the person's
        outline; ``NOT_THEM`` elsewhere. And ``(N,)`` the mean probability of the kept labels,
        NaN where nothing was kept."""
        count, (height, width) = len(frames), frames[0].shape[:2]
        labels_out = np.full((count, height, width), NOT_THEM, np.uint8)
        certainty = np.full(count, np.nan)
        margin = self.person_margin
        jobs = [(i, person_crop(boxes[i, :4], margin)) for i in valid]
        batch_size = self.batch
        for start in range(0, len(jobs), batch_size):
            batch = jobs[start : start + batch_size]
            # Cropped on the GPU and resized there by the processor; every batch the same size,
            # so the compiled model compiles once.
            images = [padded_on(frames[i], crop, self.device) for i, crop in batch]
            images += [images[-1]] * (batch_size - len(images))
            inputs = self.processor(images=images, return_tensors="pt", device=self.device)
            inputs = inputs.to(self.device, torch.bfloat16)
            logits = self.model(**inputs).logits.float()[: len(batch)]
            for (i, (x0, y0, x1, y1)), logit in zip(batch, logits, strict=True):
                logit = F.interpolate(logit[None], (y1 - y0, x1 - x0), mode="bilinear")
                probability, label = logit.softmax(dim=1)[0].max(dim=0)
                fx0, fy0, fx1, fy1 = max(x0, 0), max(y0, 0), min(x1, width), min(y1, height)
                window = (slice(fy0 - y0, fy1 - y0), slice(fx0 - x0, fx1 - x0))
                labels = label[window].byte().cpu().numpy()
                box = boxes[i, :4] - np.array([fx0, fy0, fx0, fy0])
                inside = outline(labels != 0, box, margin / 2)
                labels_out[i, fy0:fy1, fx0:fx1][inside] = labels[inside]
                if inside.any():
                    kept = torch.from_numpy(inside).to(probability.device)
                    certainty[i] = float(probability[window][kept].mean())
        return labels_out, certainty


def zoom_cuts(keypoints: np.ndarray, min_confidence: float) -> np.ndarray:
    """``(N,)`` bool: the frames of a piece, given its ``(N, 17, 3)`` COCO keypoints, where the
    face's size (the spread of its confident nose, eye and ear points, three at least) changes by
    ``ZOOM_JUMP`` or more since the last frame that showed it: a cut to or from a close-up.
    Motion is not smooth across these, though the shot detection saw one shot."""
    cuts = np.zeros(len(keypoints), bool)
    last = np.nan
    for i, face in enumerate(keypoints[:, :5]):
        seen = face[face[:, 2] >= min_confidence, :2]
        if len(seen) < 3:
            continue
        size = np.sqrt(((seen - seen.mean(axis=0)) ** 2).sum(axis=1).mean())
        cuts[i] = np.isfinite(last) and not 1 / ZOOM_JUMP < size / last < ZOOM_JUMP
        last = size
    return cuts


def follow(found: list[np.ndarray], select: int, min_iou: float) -> np.ndarray:
    """``(N, 5)`` one person's box and score per frame, NaN where they were lost.

    They are the largest person in frame ``select``; from there, forwards and backwards, each
    frame's box is the one overlapping theirs in the last frame they were found in the most, if
    by at least ``min_iou``.

    Args:
        found: per frame, ``(K, 5)`` boxes as x0, y0, x1, y1 and score.
    """
    out = np.full((len(found), 5), np.nan)
    first = found[select]
    out[select] = first[np.argmax((first[:, 2] - first[:, 0]) * (first[:, 3] - first[:, 1]))]
    for order in (range(select + 1, len(found)), range(select - 1, -1, -1)):
        last = out[select]
        for i in order:
            if not len(found[i]):
                continue
            overlaps = box_iou(last[:4], found[i][:, :4])
            if overlaps.max() >= min_iou:
                out[i] = last = found[i][overlaps.argmax()]
    return out


def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """``(K,)`` intersection over union of a ``(4,)`` box with ``(K, 4)`` boxes."""
    x0 = np.maximum(box[0], boxes[:, 0])
    y0 = np.maximum(box[1], boxes[:, 1])
    x1 = np.minimum(box[2], boxes[:, 2])
    y1 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)

    def area(b):
        return (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])

    return inter / np.maximum(area(box) + area(boxes) - inter, 1e-9)


def outline(body: np.ndarray, box: np.ndarray, margin: float) -> np.ndarray:
    """``(H, W)`` bool: the connected regions of ``body`` with at least half their pixels inside
    ``box`` (x0, y0, x1, y1), widened by ``margin`` times its larger side."""
    x0, y0, x1, y1 = box
    pad = margin * max(x1 - x0, y1 - y0)
    inside = np.zeros_like(body)
    rows = slice(max(int(y0 - pad), 0), max(int(np.ceil(y1 + pad)), 0))
    columns = slice(max(int(x0 - pad), 0), max(int(np.ceil(x1 + pad)), 0))
    inside[rows, columns] = True
    count, regions = cv2.connectedComponents(body.astype(np.uint8))
    sizes = np.bincount(regions.ravel(), minlength=count)
    within = np.bincount(regions[inside], minlength=count)
    kept = within * 2 >= sizes
    kept[0] = False  # the background
    return kept[regions]


def face_box(points, span, min_keypoint, size):
    """x0, y0, x1, y1 of a square ``span`` times the confident face keypoints' extent, around
    them, from ``(17, 3)`` COCO keypoints; None without three confident ones."""
    face = points[:5]
    face = face[face[:, 2] >= min_keypoint]
    if len(face) < 3:
        return None
    centre = face[:, :2].mean(axis=0)
    half = span * max(np.ptp(face[:, 0]), np.ptp(face[:, 1]), 8.0)
    x0, y0 = np.maximum(np.floor(centre - half), 0).astype(int)
    x1, y1 = np.minimum(np.ceil(centre + half), size).astype(int)
    return (x0, y0, x1, y1) if x1 - x0 >= 16 and y1 - y0 >= 16 else None


def person_crop(box, margin):
    """x0, y0, x1, y1 of a 3:4 crop around a ``(4,)`` box, possibly reaching outside the frame."""
    x0, y0, x1, y1 = box
    pad = margin * max(x1 - x0, y1 - y0)
    x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    height = max(y1 - y0, (x1 - x0) * INPUT_HEIGHT / INPUT_WIDTH)
    width = height * INPUT_WIDTH / INPUT_HEIGHT
    return (
        round(cx - width / 2),
        round(cy - height / 2),
        round(cx + width / 2),
        round(cy + height / 2),
    )


def padded_on(frame: np.ndarray, crop, device) -> torch.Tensor:
    """``(3, h, w)`` uint8 tensor on ``device``: the crop's pixels, black where it leaves the
    frame."""
    x0, y0, x1, y1 = crop
    height, width = frame.shape[:2]
    fx0, fy0, fx1, fy1 = max(x0, 0), max(y0, 0), min(x1, width), min(y1, height)
    out = torch.zeros((3, y1 - y0, x1 - x0), dtype=torch.uint8, device=device)
    if fx1 > fx0 and fy1 > fy0:
        window = torch.from_numpy(np.ascontiguousarray(frame[fy0:fy1, fx0:fx1])).to(device)
        out[:, fy0 - y0 : fy1 - y0, fx0 - x0 : fx1 - x0] = window.permute(2, 0, 1)
    return out


def _loops(connections):
    """The closed loops of landmark indices a set of edges forms."""
    neighbours = {}
    for edge in connections:
        neighbours.setdefault(edge.start, []).append(edge.end)
        neighbours.setdefault(edge.end, []).append(edge.start)
    unvisited, found = set(neighbours), []
    while unvisited:
        start = unvisited.pop()
        loop, previous, current = [start], None, start
        while True:
            following = [n for n in neighbours[current] if n != previous]
            if not following or following[0] == start:
                break
            previous, current = current, following[0]
            unvisited.discard(current)
            loop.append(current)
        found.append(loop)
    return found


_C = vision.FaceLandmarksConnections
LIPS = _loops(_C.FACE_LANDMARKS_LIPS)
"""The lips' outer and inner edge."""
BROWS = (
    [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],  # one brow: upper row, then lower row back
    [300, 293, 334, 296, 336, 285, 295, 282, 283, 276],
)
EYES = _loops(_C.FACE_LANDMARKS_LEFT_EYE) + _loops(_C.FACE_LANDMARKS_RIGHT_EYE)
IRISES = _loops(_C.FACE_LANDMARKS_LEFT_IRIS) + _loops(_C.FACE_LANDMARKS_RIGHT_IRIS)
NOSE = sorted({e.start for e in _C.FACE_LANDMARKS_NOSE} | {e.end for e in _C.FACE_LANDMARKS_NOSE})


def face_parts_of(points: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """``(H, W)`` uint8 face part labels (``FACE_PART_NAMES``) filled from ``(478, 3)``
    landmarks: nose, brows, eyes, irises, lips, then the mouth opening over them."""
    labels = np.zeros(size, np.uint8)
    xy = points[:, :2].astype(np.float32)

    def fill(indices, value):
        cv2.fillPoly(labels, [np.round(xy[indices]).astype(np.int32)], value)

    hull = cv2.convexHull(xy[NOSE], returnPoints=False)[:, 0]
    fill([NOSE[k] for k in hull], 1)
    for brow in BROWS:
        fill(brow, 2)
    for eye in EYES:
        fill(eye, 3)
    for iris in IRISES:
        fill(iris, 4)
    outer, inner = sorted(LIPS, key=lambda loop: -cv2.contourArea(xy[loop]))
    fill(outer, 5)
    fill(inner, 6)
    return labels


@contextmanager
def _timed(timings: dict[str, float], step: str):
    """Adds the block's wall-clock seconds, GPU work included, to ``timings[step]``."""
    torch.cuda.synchronize()
    started = time.perf_counter()
    yield
    torch.cuda.synchronize()
    timings[step] = timings.get(step, 0.0) + time.perf_counter() - started


SKIN_PARTS = (
    "Face_Neck", "Left_Hand", "Left_Lower_Arm", "Left_Lower_Leg", "Left_Upper_Arm",
    "Left_Upper_Leg", "Right_Hand", "Right_Lower_Arm", "Right_Lower_Leg", "Right_Upper_Arm",
    "Right_Upper_Leg", "Torso", "Left_Foot", "Right_Foot",
)  # fmt: skip
"""Sapiens2's labels of bare skin, the colour reference ``skin_coloured`` compares with."""
MIN_SKIN_SPREAD = 0.01
"""Least standard deviation of the skin's chromaticity: an even reference is not too strict."""
MIN_SKIN_SHARE = 0.9
"""Share of a region's pixels that must be skin-coloured for ``skin_coloured``: bare skin has
over 93 %, a leopard print whose median hue is the skin's 66 %."""
SKIN_TEXTURE = 4.0
"""How many times the skin's local contrast a region's may have for ``skin_coloured``: bare
skin is smooth, patterned fabric is not (a leopard print: about 20 times)."""
TEXTURE_WINDOW = 5
"""Pixels: the side of the window local contrast is measured in."""


def skin_coloured(
    frame: np.ndarray, labels: np.ndarray, regions: np.ndarray, max_distance: float
) -> np.ndarray:
    """``(H, W)`` bool: the connected parts of ``regions`` whose colour is the skin's.

    Sapiens2 labels bare legs as clothing now and then. The skin is what it labels as skin in
    the same frame (``SKIN_PARTS``); colours are compared by chromaticity (each channel over
    their sum), which shading does not change. A region is skin where its median and at least
    ``MIN_SKIN_SHARE`` of its pixels lie within ``max_distance`` standard deviations of the
    skin's mean (Mahalanobis; the spread at least ``MIN_SKIN_SPREAD``), and it is as smooth as
    the skin: its median local contrast (the brightness's standard deviation over its mean, in
    ``TEXTURE_WINDOW`` windows) at most ``SKIN_TEXTURE`` times the skin's. Colour alone took a
    leopard print for skin: its tan base has the skin's hue.

    Args:
        frame: ``(H, W, 3)`` uint8 RGB.
        labels: ``(H, W)`` Sapiens2's labels, indices into ``PART_NAMES``.
        regions: ``(H, W)`` bool, e.g. the pixels labelled as clothing.
    """
    out = np.zeros_like(regions)
    skin = np.isin(labels, [PART_NAMES.index(name) for name in SKIN_PARTS])
    if skin.sum() < 50 or not regions.any():
        return out
    colour = frame.astype(np.float32) + 1
    chroma = colour[..., :2] / colour.sum(axis=-1, keepdims=True)  # r, g chromaticity
    reference = chroma[skin]
    mean = reference.mean(axis=0)
    inverse = np.linalg.inv(np.cov(reference.T) + MIN_SKIN_SPREAD**2 * np.eye(2))
    offset = chroma - mean
    distance = np.sqrt(np.einsum("...i,ij,...j->...", offset, inverse, offset))
    grey = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
    window = (TEXTURE_WINDOW, TEXTURE_WINDOW)
    local_mean = cv2.blur(grey, window)
    spread = np.sqrt(np.maximum(cv2.blur(grey**2, window) - local_mean**2, 0))
    contrast = spread / (local_mean + 10)  # + 10: dark pixels' noise is no texture
    smooth = SKIN_TEXTURE * max(float(np.median(contrast[skin])), 1e-3)
    count, parts = cv2.connectedComponents(regions.astype(np.uint8))
    for k in range(1, count):
        inside = parts == k
        difference = np.median(chroma[inside], axis=0) - mean
        if (
            np.sqrt(difference @ inverse @ difference) <= max_distance
            and (distance[inside] <= max_distance).mean() >= MIN_SKIN_SHARE
            and np.median(contrast[inside]) <= smooth
        ):
            out |= inside
    return out
