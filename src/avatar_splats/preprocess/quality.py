"""How good a frame of a person is to build an avatar from: one score, and the gates it must pass.

A frame is measured (``measure_frames``) on what the cheap models found, the box and COCO
keypoints, and its pixels:

- resolution: the person's height in the source video's pixels, the detail there is to get,
- sharpness: ``triage.focus`` over their body,
- light: how bright their body is (median luminance, with blown-out pixels against it) and how
  evenly lit (the 10th over the 90th percentile of its luminance, blurred so a print is no
  shadow: hard shadows pull it down),
- pose: how far their joints are from an A-pose (``A_POSE``), facing the camera or turned away,
  in torso lengths after the best scale and shift,
- occlusion: the share of their torso, head and thighs their forearms and hands (and a shin
  crossing the other thigh) cover,
- whole: their body joints all placed and the box clear of the frame's edge,
- alone: how little of their box another person's covers.

The body's pixels are drawn from the keypoints (``body_mask``): torso, head and limbs as
polygons and capsules, a few per cent off the outline but with no segmentation model. Light
measured there cannot tell dark skin from dim light; ``dark`` and ``bright`` are set for the
footage at hand.

``score_frames`` maps the measures to factors in [0, 1], their weighted geometric mean is the
score, and each factor has a gate: a frame below any gate is rejected, with the factors below
theirs as the reasons. Resolution and sharpness weigh more than pose: a frame with a slightly
worse pose but more detail is the better one.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from tqdm.auto import tqdm

from avatar_splats.preprocess.triage import focus

FACTOR_NAMES = (
    "whole", "resolution", "sharpness", "brightness", "evenness", "pose", "occlusion", "alone",
    "identity",
)  # fmt: skip
FACE_FACTOR_NAMES = (
    "found", "resolution", "sharpness", "brightness", "evenness", "occlusion", "alone",
    "identity",
)  # fmt: skip
FACE_SPAN = 1.6
"""A face's height over the largest distance between its COCO keypoints (nose, eyes, ears)."""
FACE_WIDTH = 0.8
"""A face's width over its height, for the ellipse drawn from keypoints."""
BODY_JOINTS = tuple(range(5, 17))
"""COCO's shoulders, elbows, wrists, hips, knees and ankles."""
A_POSE = np.array(
    [
        (0.40, -1.00), (-0.40, -1.00),  # left, right shoulder
        (0.79, -0.54), (-0.79, -0.54),  # elbows: upper arms 40 degrees out from the body
        (1.11, -0.16), (-1.11, -0.16),  # wrists
        (0.18, 0.00), (-0.18, 0.00),  # hips
        (0.33, 0.84), (-0.33, 0.84),  # knees: legs 10 degrees apart
        (0.48, 1.68), (-0.48, 1.68),  # ankles
    ]
)  # fmt: skip
"""``(12, 2)`` the ``BODY_JOINTS`` of an A-pose facing the camera, in torso lengths (the
shoulders' middle to the hips'), x right, y down, from the hips' middle; her left on the
image's right."""
MIRRORED = (1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10)
"""``BODY_JOINTS``' order with left and right swapped: seen from behind."""
LIMBS = {
    "upper arm": ((5, 7), (6, 8), 0.10),
    "forearm": ((7, 9), (8, 10), 0.08),
    "thigh": ((11, 13), (12, 14), 0.15),
    "shin": ((13, 15), (14, 16), 0.10),
}
"""Per limb, its left and right COCO segment and its radius in torso lengths."""
HEAD_RADIUS = 0.28
"""The head's radius in torso lengths, around the face keypoints' middle."""
HAND = (0.35, 0.10)
"""A hand's centre past the wrist, in forearm lengths, and its radius in torso lengths."""
LIGHT_BLUR = 0.08
"""Torso lengths: the Gaussian evenness is measured after (``lighting``)."""
CROP_MARGIN = 0.15
"""Share of the box's larger side ``measure_frames`` crops around it: limbs reach past the
detector's box now and then."""
MASK_SIDE = 256
"""Pixels: the longer side of the box the body is drawn in for occlusion."""


def torso_length(points: np.ndarray) -> float:
    """Pixels from the shoulders' middle to the hips' middle of ``(17, 2)`` keypoints."""
    return float(np.linalg.norm(points[[5, 6]].mean(0) - points[[11, 12]].mean(0)))


def a_pose_distance(keypoints: np.ndarray, min_confidence: float, min_joints: int = 10) -> float:
    """Mean distance, in torso lengths, of ``(17, 3)`` keypoints' placed ``BODY_JOINTS`` from
    ``A_POSE`` (or it mirrored, whichever is nearer) after the best scale and shift; no rotation:
    upright is part of the pose. NaN with fewer than ``min_joints`` placed."""
    sure = np.nan_to_num(keypoints[list(BODY_JOINTS), 2]) >= min_confidence
    if sure.sum() < min_joints:
        return float("nan")
    seen = keypoints[list(BODY_JOINTS), :2][sure]
    best = np.inf
    for template in (A_POSE, A_POSE[list(MIRRORED)]):
        model = template[sure]
        centred = model - model.mean(0)
        scale = (centred * (seen - seen.mean(0))).sum() / (centred**2).sum()
        if scale <= 0:
            continue
        fitted = scale * centred + seen.mean(0)
        best = min(best, float(np.linalg.norm(fitted - seen, axis=1).mean() / scale))
    return best if np.isfinite(best) else float("nan")


def _capsule(canvas: np.ndarray, a: np.ndarray, b: np.ndarray, radius: float) -> None:
    thickness = max(1, round(2 * radius))
    cv2.line(canvas, np.round(a).astype(int), np.round(b).astype(int), 1, thickness)


def _hands(points: np.ndarray, sure: np.ndarray, torso: float):
    """``(centre, radius)`` of each hand whose elbow and wrist are placed."""
    for elbow, wrist in ((7, 9), (8, 10)):
        if sure[elbow] and sure[wrist]:
            reach = points[wrist] - points[elbow]
            yield points[wrist] + HAND[0] * reach, HAND[1] * torso


def body_mask(keypoints: np.ndarray, shape: tuple[int, int], min_confidence: float) -> np.ndarray:
    """``(H, W)`` bool: a body drawn from ``(17, 3)`` keypoints in an image of ``shape``: the
    torso's polygon, the head and the placed limbs as capsules (``LIMBS``), the hands; all False
    without both shoulders and hips."""
    canvas = np.zeros(shape, np.uint8)
    points, sure = keypoints[:, :2], np.nan_to_num(keypoints[:, 2]) >= min_confidence
    if not sure[[5, 6, 11, 12]].all():
        return canvas.astype(bool)
    torso = torso_length(points)
    cv2.fillPoly(canvas, [np.round(points[[5, 6, 12, 11]]).astype(np.int32)], 1)
    for left, right, radius in LIMBS.values():
        for a, b in (left, right):
            if sure[a] and sure[b]:
                _capsule(canvas, points[a], points[b], radius * torso)
    face = sure[:5]
    if face.any():
        centre = np.round(points[:5][face].mean(0)).astype(int)
        cv2.circle(canvas, centre, round(HEAD_RADIUS * torso), 1, -1)
        _capsule(canvas, centre, points[[5, 6]].mean(0), 0.12 * torso)  # the neck
    for centre, radius in _hands(points, sure, torso):
        cv2.circle(canvas, np.round(centre).astype(int), max(1, round(radius)), 1, -1)
    return canvas.astype(bool)


def occlusion(keypoints: np.ndarray, min_confidence: float) -> float:
    """The share of the torso, head and thighs drawn from ``(17, 3)`` keypoints that forearms
    and hands, or a shin across the other thigh, cover. NaN without both shoulders and hips."""
    points, sure = keypoints[:, :2].copy(), np.nan_to_num(keypoints[:, 2]) >= min_confidence
    if not sure[[5, 6, 11, 12]].all():
        return float("nan")
    placed = points[sure]
    low, high = placed.min(0), placed.max(0)
    factor = MASK_SIDE / max(float((high - low).max()), 1.0)
    points = (points - low) * factor + MASK_SIDE * 0.25
    torso = torso_length(points)
    size = (int(MASK_SIDE * 1.5),) * 2
    covered, covering = np.zeros(size, np.uint8), np.zeros(size, np.uint8)
    cv2.fillPoly(covered, [np.round(points[[5, 6, 12, 11]]).astype(np.int32)], 1)
    if sure[:5].any():
        centre = np.round(points[:5][sure[:5]].mean(0)).astype(int)
        cv2.circle(covered, centre, round(HEAD_RADIUS * torso), 1, -1)
    thighs = LIMBS["thigh"]
    for a, b in thighs[:2]:
        if sure[a] and sure[b]:
            _capsule(covered, points[a], points[b], thighs[2] * torso)
    for a, b in LIMBS["forearm"][:2]:
        if sure[a] and sure[b]:
            _capsule(covering, points[a], points[b], LIMBS["forearm"][2] * torso)
    for centre, radius in _hands(points, sure, torso):
        cv2.circle(covering, np.round(centre).astype(int), max(1, round(radius)), 1, -1)
    # a shin across the other leg's thigh
    shin_radius = LIMBS["shin"][2] * torso
    for (a, b), (c, d) in zip(LIMBS["shin"][:2], reversed(thighs[:2]), strict=True):
        if sure[a] and sure[b] and sure[c] and sure[d]:
            other = np.zeros(size, np.uint8)
            _capsule(other, points[c], points[d], thighs[2] * torso)
            shin = np.zeros(size, np.uint8)
            _capsule(shin, points[a], points[b], shin_radius)
            covering |= shin & other
    area = covered.sum()
    return float((covered & covering).sum() / area) if area else float("nan")


def lighting(frame: np.ndarray, mask: np.ndarray, blur: float = 0.0) -> tuple[float, float, float]:
    """Of the ``(H, W)`` bool ``mask``'s pixels of an RGB uint8 frame: the median luminance in
    [0, 1], the share blown out (above 0.97), and the 10th percentile of luminance over the
    90th (1: evenly lit), that of the luminance blurred over the mask by a Gaussian of ``blur``
    pixels: a print or texture (a leopard swimsuit's spots) is not a shadow, which is larger.
    NaN where the mask is empty."""
    if not mask.any():
        return (float("nan"),) * 3
    grey = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    values = grey[mask]
    middle, clipped = float(np.median(values)), float((values > 0.97).mean())
    if blur > 0:
        weight = mask.astype(np.float32)
        smooth = cv2.GaussianBlur(grey * weight, (0, 0), blur)
        smooth /= np.maximum(cv2.GaussianBlur(weight, (0, 0), blur), 1e-6)
        values = smooth[mask]
    low, high = np.percentile(values, (10, 90))
    return middle, clipped, float(low / max(high, 1e-3))


def at_edge(box: np.ndarray, shape: tuple[int, int], edge: float) -> bool:
    """Whether a box (x0, y0, x1, y1) comes within ``edge`` of its side of an image's edge."""
    height, width = shape
    x0, y0, x1, y1 = box[:4]
    return bool(
        x0 <= edge * width or y0 <= edge * height or x1 >= (1 - edge) * width
        or y1 >= (1 - edge) * height
    )  # fmt: skip


@dataclass(frozen=True)
class QualitySettings:
    min_confidence: float
    """A keypoint's confidence to count as placed."""
    edge: float
    """A box this near the image's edge, in shares of its side, is cut off."""
    good_pixels: float
    """The person's height in the source video's pixels that earns full resolution."""
    good_focus: float
    """``triage.focus`` that earns full sharpness."""
    dark: float
    """Median luminance of the body (0 to 1) at which brightness is 0..."""
    bright: float
    """... and 1."""
    max_clipped: float
    """Share of the body blown out at which brightness is 0."""
    uneven: float
    """The body's 10th over 90th luminance percentile at which evenness is 0..."""
    even: float
    """... and 1."""
    pose_tolerance: float
    """Torso lengths from the A-pose at which pose is 0."""
    max_occlusion: float
    """Share of the body covered at which occlusion is 0."""
    max_overlap: float
    """Share of the box another person's covers at which alone is 0."""
    other_face: float
    """Similarity to the actor's face (``preprocess.identity``) at which identity is 0..."""
    same_face: float
    """... and 1; where no face is seen it is 1: back views are not rejected for it."""
    weights: dict[str, float]
    """Per factor its weight in the score (a weighted geometric mean); unlisted: gate only."""
    gates: dict[str, float]
    """Per factor the least it may be; unlisted: no gate."""
    min_score: float


def measure_frames(
    frames,
    boxes: np.ndarray,
    keypoints: np.ndarray,
    source_scale: float,
    overlap: np.ndarray,
    settings: QualitySettings,
    progress: bool = False,
) -> dict[str, np.ndarray]:
    """Raw measures of N frames' person: ``(N,)`` each of ``pixels`` (height in source pixels),
    ``focus``, ``brightness``, ``clipped``, ``evenness``, ``pose`` (``a_pose_distance``),
    ``occlusion``, ``joints`` (body joints placed), ``cut`` (box at the edge) and ``overlap``;
    NaN where nobody was found.

    Args:
        frames: N ``(H, W, 3)`` RGB uint8 frames.
        boxes: ``(N, 4+)`` x0, y0, x1, y1 in their pixels; NaN where nobody.
        keypoints: ``(N, 17, 3)`` COCO keypoints in their pixels and confidence.
        source_scale: the frames' size over the source video's.
        overlap: ``(N,)`` the largest share of the box another person's covers.
        progress: show a bar.
    """
    count = len(frames)
    names = ("pixels", "focus", "brightness", "clipped", "evenness", "pose", "occlusion")
    out = {name: np.full(count, np.nan) for name in names}
    out["joints"] = np.zeros(count)
    out["cut"] = np.zeros(count, bool)
    out["overlap"] = np.asarray(overlap, float)
    for i in tqdm(range(count), desc="quality", unit="frame", disable=not progress, leave=False):
        box = boxes[i, :4]
        if not np.isfinite(box).all():
            continue
        shape = frames[i].shape[:2]
        points = keypoints[i]
        # everything but the edge test on a crop around the box: a few times faster than the
        # whole frame, and the body drawn from the keypoints lies in it
        x0, y0, x1, y1 = box
        pad = CROP_MARGIN * max(x1 - x0, y1 - y0)
        cx0, cy0 = max(int(x0 - pad), 0), max(int(y0 - pad), 0)
        cx1, cy1 = min(int(np.ceil(x1 + pad)), shape[1]), min(int(np.ceil(y1 + pad)), shape[0])
        crop = frames[i][cy0:cy1, cx0:cx1]
        shift = np.array([cx0, cy0, 0.0])
        mask = body_mask(points - shift, crop.shape[:2], settings.min_confidence)
        out["pixels"][i] = (y1 - y0) / source_scale
        out["focus"][i] = focus(crop, box - shift[[0, 1, 0, 1]], mask if mask.any() else None)
        blur = LIGHT_BLUR * torso_length(points[:, :2]) if mask.any() else 0.0
        brightness, clipped, evenness = lighting(crop, mask, blur)
        out["brightness"][i], out["clipped"][i], out["evenness"][i] = brightness, clipped, evenness
        out["pose"][i] = a_pose_distance(points, settings.min_confidence)
        out["occlusion"][i] = occlusion(points, settings.min_confidence)
        out["joints"][i] = (
            np.nan_to_num(points[list(BODY_JOINTS), 2]) >= settings.min_confidence
        ).sum()
        out["cut"][i] = at_edge(box, shape, settings.edge)
    return out


def _ramp(values: np.ndarray, zero: float, one: float) -> np.ndarray:
    return np.clip(np.nan_to_num((values - zero) / (one - zero), nan=0.0), 0, 1)


def score_frames(
    measures: dict[str, np.ndarray], settings: QualitySettings
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, list[str]]:
    """Per frame of ``measure_frames``: the ``FACTOR_NAMES`` in [0, 1], the score, whether it
    passes (every gate and ``min_score``), and why not (the factors below their gates, or the
    score); empty where it passes."""
    s = settings
    whole = measures["joints"] / len(BODY_JOINTS) * np.where(measures["cut"], 0.5, 1.0)
    factors = {
        "whole": whole,
        "resolution": _ramp(measures["pixels"], 0, s.good_pixels),
        "sharpness": _ramp(measures["focus"], 0, s.good_focus),
        "brightness": _ramp(measures["brightness"], s.dark, s.bright)
        * _ramp(measures["clipped"], s.max_clipped, 0),
        "evenness": _ramp(measures["evenness"], s.uneven, s.even),
        "pose": _ramp(measures["pose"], s.pose_tolerance, 0),
        "occlusion": _ramp(measures["occlusion"], s.max_occlusion, 0),
        "alone": _ramp(np.nan_to_num(measures["overlap"], nan=1.0), s.max_overlap, 0),
        "identity": _identity(measures, s),
    }
    measured_by = {
        "resolution": "pixels", "sharpness": "focus", "brightness": "brightness",
        "evenness": "evenness", "pose": "pose", "occlusion": "occlusion",
    }  # fmt: skip
    return (factors, *_verdicts(factors, measures, measured_by, s))


def _identity(measures: dict[str, np.ndarray], s) -> np.ndarray:
    """Identity in [0, 1] from ``measures["identity"]`` (similarity to the actor's face), 1
    where it is unknown or not given."""
    similarity = measures.get("identity")
    if similarity is None:
        return np.ones(len(measures["overlap"]))
    return np.where(np.isfinite(similarity), _ramp(similarity, s.other_face, s.same_face), 1.0)


def _verdicts(factors, measures, measured_by, s) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """The score (the weighted geometric mean of ``factors``), whether each frame passes and
    why not, for ``score_frames`` and ``score_faces``."""
    total = sum(s.weights.values())
    logs = sum(w * np.log(np.maximum(factors[name], 1e-6)) for name, w in s.weights.items())
    score = np.exp(logs / total)
    reasons = []
    for i in range(len(score)):
        below = [name for name, gate in s.gates.items() if factors[name][i] < gate]
        # a factor its frame could not measure (no torso, too few joints) is no reason of its
        # own: the gate that failed for the same cause, such as whole, says why
        failed = [
            f"{name} {factors[name][i]:.2f}"
            for name in below
            if name not in measured_by or np.isfinite(measures[measured_by[name]][i])
        ]
        if below and not failed:
            failed = ["not measurable: " + ", ".join(below)]
        if not failed and score[i] < s.min_score:
            failed = [f"score {score[i]:.2f}"]
        reasons.append(", ".join(failed))
    passed = np.array([not reason for reason in reasons])
    return score, passed, reasons


@dataclass(frozen=True)
class FaceQualitySettings:
    """``QualitySettings``' counterparts for a face: how good a frame is for the head."""

    min_confidence: float
    edge: float
    good_pixels: float
    """The face's height in the source's pixels that earns full resolution."""
    good_focus: float
    dark: float
    bright: float
    max_clipped: float
    uneven: float
    even: float
    max_occlusion: float
    """Share of the face hands and forearms cover at which occlusion is 0."""
    max_overlap: float
    other_face: float
    same_face: float
    weights: dict[str, float]
    gates: dict[str, float]
    min_score: float


def face_outline(
    keypoints: np.ndarray, landmarks: np.ndarray | None, min_confidence: float
) -> np.ndarray | None:
    """``(K, 2)`` a face's outline: the convex hull of ``(478, 3)`` MediaPipe landmarks where
    they are finite, else an ellipse around ``(17, 3)`` COCO keypoints' face (at least three of
    nose, eyes and ears placed, ``FACE_SPAN``); None where neither is."""
    if landmarks is not None and np.isfinite(landmarks).all():
        return cv2.convexHull(landmarks[:, :2].astype(np.float32))[:, 0]
    sure = np.nan_to_num(keypoints[:5, 2]) >= min_confidence
    if sure.sum() < 3:
        return None
    points = keypoints[:5, :2][sure]
    spread = max(np.linalg.norm(a - b) for a in points for b in points)
    height = FACE_SPAN * spread
    centre = points.mean(0)
    axes = (max(1, round(FACE_WIDTH * height / 2)), max(1, round(height / 2)))
    return cv2.ellipse2Poly(np.round(centre).astype(int), axes, 0, 0, 360, 10).astype(np.float32)


def _face_covering(
    shape, keypoints, hands, height: float, min_confidence: float, shift: np.ndarray
) -> np.ndarray:
    """``(H, W)`` uint8: hands (MediaPipe's, else from the keypoints) and forearms that could
    cover a face ``height`` pixels tall, in a crop ``shift`` from the frame's origin."""
    covering = np.zeros(shape, np.uint8)
    points = keypoints[:, :2] - shift
    sure = np.nan_to_num(keypoints[:, 2]) >= min_confidence
    for a, b in LIMBS["forearm"][:2]:
        if sure[a] and sure[b]:
            _capsule(covering, points[a], points[b], 0.15 * height)
    found = False
    if hands is not None:
        for hand in hands:
            if np.isfinite(hand[:, :2]).all():
                hull = cv2.convexHull((hand[:, :2] - shift).astype(np.float32))[:, 0]
                cv2.fillPoly(covering, [np.round(hull).astype(np.int32)], 1)
                found = True
    if not found:
        for elbow, wrist in ((7, 9), (8, 10)):
            if sure[elbow] and sure[wrist]:
                centre = points[wrist] + HAND[0] * (points[wrist] - points[elbow])
                cv2.circle(
                    covering, np.round(centre).astype(int), max(1, round(0.3 * height)), 1, -1
                )
    return covering


def measure_faces(
    frames,
    keypoints: np.ndarray,
    face_landmarks: np.ndarray | None,
    hands: np.ndarray | None,
    source_scale: float,
    overlap: np.ndarray,
    settings: FaceQualitySettings,
    progress: bool = False,
) -> dict[str, np.ndarray]:
    """Raw measures of N frames' face (``face_outline``): ``(N,)`` each of ``pixels`` (its
    height in source pixels), ``focus``, ``brightness``, ``clipped``, ``evenness``,
    ``occlusion`` (the share hands and forearms cover), ``found``, ``cut`` (at the frame's
    edge) and ``overlap``; NaN where no face was found.

    Args:
        face_landmarks: ``(N, 478, 3)`` MediaPipe's, or None (the keypoints place the face).
        hands: ``(N, 2, 21, 3)`` MediaPipe's, or None (the keypoints place the hands).
        others: as ``measure_frames``.
    """
    count = len(frames)
    names = ("pixels", "focus", "brightness", "clipped", "evenness", "occlusion")
    out = {name: np.full(count, np.nan) for name in names}
    out["found"] = np.zeros(count, bool)
    out["cut"] = np.zeros(count, bool)
    out["overlap"] = np.asarray(overlap, float)
    for i in tqdm(range(count), desc="faces", unit="frame", disable=not progress, leave=False):
        landmarks = None if face_landmarks is None else face_landmarks[i]
        outline = face_outline(keypoints[i], landmarks, settings.min_confidence)
        if outline is None:
            continue
        shape = frames[i].shape[:2]
        low, high = outline.min(0), outline.max(0)
        height = float(high[1] - low[1])
        pad = 0.3 * height
        x0, y0 = max(int(low[0] - pad), 0), max(int(low[1] - pad), 0)
        x1 = min(int(np.ceil(high[0] + pad)), shape[1])
        y1 = min(int(np.ceil(high[1] + pad)), shape[0])
        if x1 <= x0 or y1 <= y0:
            continue
        crop = frames[i][y0:y1, x0:x1]
        shift = np.array([x0, y0], np.float32)
        mask = np.zeros(crop.shape[:2], np.uint8)
        cv2.fillPoly(mask, [np.round(outline - shift).astype(np.int32)], 1)
        mask = mask.astype(bool)
        if not mask.any():
            continue
        box = np.array([*(low - shift), *(high - shift)])
        hand = None if hands is None else hands[i]
        covering = _face_covering(
            crop.shape[:2], keypoints[i], hand, height, settings.min_confidence, shift
        )
        out["found"][i] = True
        out["pixels"][i] = height / source_scale
        out["focus"][i] = focus(crop, box, mask)
        brightness, clipped, evenness = lighting(crop, mask, 0.05 * height)
        out["brightness"][i], out["clipped"][i], out["evenness"][i] = brightness, clipped, evenness
        out["occlusion"][i] = float((covering.astype(bool) & mask).sum() / mask.sum())
        out["cut"][i] = at_edge(np.array([*low, *high]), shape, settings.edge)
    return out


def score_faces(
    measures: dict[str, np.ndarray], settings: FaceQualitySettings
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, list[str]]:
    """``score_frames`` for faces: the ``FACE_FACTOR_NAMES``, the score, whether each passes and
    why not."""
    s = settings
    factors = {
        "found": np.where(measures["found"], np.where(measures["cut"], 0.5, 1.0), 0.0),
        "resolution": _ramp(measures["pixels"], 0, s.good_pixels),
        "sharpness": _ramp(measures["focus"], 0, s.good_focus),
        "brightness": _ramp(measures["brightness"], s.dark, s.bright)
        * _ramp(measures["clipped"], s.max_clipped, 0),
        "evenness": _ramp(measures["evenness"], s.uneven, s.even),
        "occlusion": _ramp(measures["occlusion"], s.max_occlusion, 0),
        "alone": _ramp(np.nan_to_num(measures["overlap"], nan=1.0), s.max_overlap, 0),
        "identity": _identity(measures, s),
    }
    measured_by = {
        "resolution": "pixels", "sharpness": "focus", "brightness": "brightness",
        "evenness": "evenness", "occlusion": "occlusion",
    }  # fmt: skip
    return (factors, *_verdicts(factors, measures, measured_by, s))
