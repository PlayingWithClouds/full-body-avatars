"""Triage: a quick pass over a whole video that scores how usable each moment is for
reconstructing its main person, and groups the moments into shots.

Built to be cheap:

1. Sample cheaply: only the video's keyframes are decoded (every 2-4 s in typical files), or
   every ``every_n``-th frame.
2. Cheapest model first: YOLO-pose finds people and 17 joints. Samples with nobody in them, or
   someone too small or too hidden, stop there.
3. Expensive check last: only the samples that pass get Sapiens2's body parts at half
   resolution, to see whether another body touches the main person; the other person is often
   barely in frame (no head, just a leg) and no person detector finds them, but their skin is
   still labelled as a body part.

The main person is the largest one in each sample. Each sample gets a score from 0 to 1, the
product of ``FACTORS``; neighbouring samples that look alike (colour histograms) form shots.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm

from avatar_splats.data.video import open_video
from avatar_splats.records import ArrayRecord

FACTORS = ("size", "joints", "whole", "sharpness", "alone")
"""What a sample's score multiplies, each in [0, 1]: the main person's height, the joints seen,
not cut off by the frame's edge, sharpness against the video's median, and no other body
touching them."""
CLOTHING = (1, 2, 9, 10, 13, 18, 19, 23)
"""Sapiens2's labels that are no body: apparel, eyeglass, shoes, socks, lower and upper
clothing."""
GREY = (124, 116, 104)
"""Sapiens2's mean colour, for padding: reads as nothing."""
FOCUS_TILE = 16
"""Pixels: the side of the tiles ``focus`` measures, in the crop scaled to 256 pixels high."""
FOCUS_EDGES = 0.75
"""``focus`` looks at the tiles with edges: those above this quantile of gradient energy."""
FOCUS_MIN_SLOPE = 400.0
"""Gradient energy (Sobel, grey levels²) a tile needs to count as an edge, about 2.5 grey levels
per pixel: below it ``focus`` would measure pixel noise and compression, which look sharp."""
FOCUS_QUANTILE = 0.9
"""Which of those tiles' sharpness ``focus`` takes: the sharpest few, but not one stray edge."""


def focus(image: np.ndarray, box: np.ndarray, mask: np.ndarray | None = None) -> float:
    """How sharp the sharpest edges in a box are, whatever their contrast: per ``FOCUS_TILE``
    tile of the box scaled down to 256 pixels high (never up), the Laplacian's energy over the
    gradient's, about
    one over an edge's width squared; of the tiles with edges (``FOCUS_EDGES``, at least
    ``FOCUS_MIN_SLOPE``), the ``FOCUS_QUANTILE``; 0 where no tile has an edge. Only tiles mostly
    in the ``(H, W)`` bool ``mask`` count, where given. NaN where the box is not finite or empty,
    or the mask leaves no tile.

    Motion blur and missed focus widen every edge, the sharpest too. The Laplacian's variance
    alone measured texture as much as focus: it called smooth skin before a soft background (a
    close-up with a shallow depth of field, the best a video gets) blurry next to wide shots of
    textured rock."""
    if not np.isfinite(box).all():
        return float("nan")
    x0, y0, x1, y1 = box.astype(int)
    x0, y0 = max(x0, 0), max(y0, 0)
    grey = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
    if not grey.size:
        return float("nan")
    # scaled down to 256 pixels high, never up: blowing a small face up would blur it
    factor = min(1.0, 256 / grey.shape[0])
    size = (max(1, round(grey.shape[1] * factor)), max(1, round(grey.shape[0] * factor)))
    grey = cv2.resize(grey, size, interpolation=cv2.INTER_AREA).astype(np.float32)
    laplacian = cv2.Laplacian(grey, cv2.CV_32F) ** 2
    gradient = cv2.Sobel(grey, cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(grey, cv2.CV_32F, 0, 1) ** 2
    rows, columns = (n // FOCUS_TILE for n in laplacian.shape)
    if not rows or not columns:
        return float("nan")

    def tiled(values: np.ndarray) -> np.ndarray:
        values = values[: rows * FOCUS_TILE, : columns * FOCUS_TILE]
        return values.reshape(rows, FOCUS_TILE, columns, FOCUS_TILE).swapaxes(1, 2).mean((2, 3))

    curvature, slope = tiled(laplacian).ravel(), tiled(gradient).ravel()
    if mask is not None:
        inside = cv2.resize(mask[y0:y1, x0:x1].astype(np.uint8), size, cv2.INTER_NEAREST)
        kept = tiled(inside.astype(np.float32)).ravel() >= 0.5
        curvature, slope = curvature[kept], slope[kept]
    if not slope.size:
        return float("nan")
    edges = slope >= max(float(np.quantile(slope, FOCUS_EDGES)), FOCUS_MIN_SLOPE)
    if not edges.any():
        return 0.0
    return float(np.quantile(curvature[edges] / slope[edges], FOCUS_QUANTILE))


@dataclass(frozen=True)
class TriageSettings:
    sampling: str
    """``"keyframes"`` (fast) or ``"every_n"`` (decodes every frame)."""
    every_n: int
    max_keyframe_gap: float
    """Seconds: sparser keyframes fall back to ``"every_n"``."""
    max_samples: int
    analysis_side: int
    """Pixels: samples are scaled so their longer side is this long."""
    pose_model: str
    """A YOLO-pose checkpoint (downloads on first use)."""
    parts_model: str
    """A Sapiens2 segmentation model."""
    parts_size: tuple[int, int]
    """Height and width of Sapiens2's input."""
    batch: int
    min_person: float
    """YOLO's score to count a person."""
    min_height: float
    """The main person's box height, in frame heights, to be worth anything."""
    good_height: float
    """And to count fully."""
    min_joints: int
    """Joints seen to go on to the contact check."""
    joint_seen: float
    """A joint's confidence to count as seen."""
    edge: float
    """A box this near the frame's edge, in shares of its side, counts as cut off there."""
    min_sharpness: float
    """Samples whose person is less sharp than this share of the video's median are left out
    (motion blur, focus pulls)."""
    context: float
    """The contact check looks at a crop this many times the person's box."""
    max_contact: float
    """Other body pixels per pixel of theirs that count as full contact."""
    shot_cut: float
    """Colour histogram distance (Bhattacharyya) that starts a new shot."""


@dataclass(frozen=True)
class Samples:
    """The frames a triage looks at."""

    frames: list[np.ndarray]
    """``(h, w, 3)`` uint8 RGB, scaled to ``analysis_side``."""
    times: np.ndarray
    """``(N,)`` seconds."""
    fps: float
    duration: float
    """Seconds."""
    video_size: tuple[int, int]
    """Width and height of the video."""
    sampling: str
    """How they were sampled, after any fallback."""


@dataclass(frozen=True)
class Triage(ArrayRecord):
    """Per sample of a video: what was found, and how usable it is."""

    times: np.ndarray
    """``(N,)`` seconds."""
    fps: float
    duration: float
    sample_size: np.ndarray
    """``(2,)`` width and height of the samples, which ``boxes`` and ``keypoints`` are in."""
    shot: np.ndarray
    """``(N,)`` int: the shot each sample is in, from 0, non-decreasing."""
    people: np.ndarray
    """``(N,)`` int: people found."""
    boxes: np.ndarray
    """``(N, 5)`` the main person's box as x0, y0, x1, y1 and score; NaN where nobody."""
    keypoints: np.ndarray
    """``(N, 17, 3)`` the main person's COCO joints, x, y and confidence."""
    overlap: np.ndarray
    """``(N,)`` the largest share of the main box another person's box covers."""
    sharpness: np.ndarray
    """``(N,)`` the main box's ``focus``; NaN where nobody."""
    contact: np.ndarray
    """``(N,)`` other people's body pixels per body pixel of the main person; NaN where not
    checked."""
    factors: np.ndarray
    """``(N, len(FACTORS))``."""
    score: np.ndarray
    """``(N,)`` in [0, 1]; 0 where nobody or stopped before the contact check."""


@dataclass(frozen=True)
class Shot:
    score: float  # the median of its samples' scores, 0 to 1
    begin: float  # seconds: its first sample's time
    end: float  # seconds: the next shot's first sample's time, or the video's end
    samples: np.ndarray  # (S,) indices of its samples
    best: int  # the index of its best-scoring sample


def decode_samples(
    path: Path, settings: TriageSettings, progress: bool = False, gpu: bool = False
) -> Samples:
    """The video's keyframes (or every ``every_n``-th frame where keyframes are too sparse);
    ``progress`` shows a bar in seconds of video; ``gpu`` decodes on the GPU
    (``data.video.open_video``): five times as fast for keyframes."""
    samples = _decode(path, settings, settings.sampling, progress, gpu)
    gaps = np.diff(samples.times)
    if settings.sampling == "keyframes" and (
        len(samples.frames) < 10 or np.median(gaps) > settings.max_keyframe_gap
    ):
        samples = _decode(path, settings, "every_n", progress, gpu)
    return samples


def _decode(
    path: Path, settings: TriageSettings, sampling: str, progress: bool, gpu: bool
) -> Samples:
    frames, times = [], []
    with open_video(path, gpu) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        if sampling == "keyframes":
            stream.codec_context.skip_frame = "NONKEY"
        scale = settings.analysis_side / max(stream.width, stream.height)
        size = (round(stream.width * scale) // 2 * 2, round(stream.height * scale) // 2 * 2)
        length = float(stream.duration * stream.time_base) if stream.duration else None
        bar = tqdm(
            total=round(length) if length else None, unit="s", disable=not progress,
            desc=f"decode {Path(path).name} ({sampling})", leave=False,
        )  # fmt: skip
        for index, frame in enumerate(container.decode(stream)):
            if sampling == "every_n" and index % settings.every_n:
                continue
            frames.append(
                frame.reformat(width=size[0], height=size[1], format="rgb24").to_ndarray()
            )
            times.append(float(frame.pts * stream.time_base))
            bar.update(max(0, round(times[-1]) - bar.n))
            if len(frames) >= settings.max_samples:
                break
        bar.close()
        duration = float(stream.duration * stream.time_base) if stream.duration else times[-1]
        return Samples(
            frames,
            np.array(times),
            float(stream.average_rate),
            duration,
            (stream.width, stream.height),
            sampling,
        )


def shot_numbers(frames: list[np.ndarray], shot_cut: float) -> np.ndarray:
    """``(N,)`` the shot of each sample: a new one wherever its colours (a small HSV histogram)
    differ from the previous sample's by more than ``shot_cut``."""
    histograms = []
    for frame in frames:
        small = cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
        histogram = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 4], [0, 180, 0, 256, 0, 256])
        histograms.append(cv2.normalize(histogram, None).flatten())
    change = [0.0] + [
        cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA)
        for a, b in zip(histograms, histograms[1:], strict=False)
    ]
    return np.cumsum(np.array(change) > shot_cut)


class TriageModels:
    """The triage's models, loaded once for many videos: YOLO-pose and, at first use, the
    Sapiens2 segmentation model of the contact check."""

    def __init__(self, settings: TriageSettings, device: str):
        from ultralytics import YOLO

        self.settings, self.device = settings, device
        self.pose = YOLO(settings.pose_model)
        self._parts = None

    @property
    def parts(self):
        """``(processor, model)`` of ``settings.parts_model``, loaded at first use."""
        if self._parts is None:
            from transformers import AutoImageProcessor, Sapiens2ForSemanticSegmentation

            model_id = self.settings.parts_model
            model = Sapiens2ForSemanticSegmentation.from_pretrained(model_id, dtype=torch.bfloat16)
            self._parts = (
                AutoImageProcessor.from_pretrained(model_id),
                model.to(self.device).eval(),
            )
        return self._parts


def triage(
    samples: Samples,
    settings: TriageSettings,
    device: str,
    prefilter: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    progress: bool = False,
    models: TriageModels | None = None,
) -> tuple[Triage, dict[int, tuple[list[int], np.ndarray]]]:
    """Score every sample.

    ``prefilter`` takes the main person's ``(N, 5)`` boxes and ``(N, 17, 3)`` keypoints (NaN
    where nobody) and returns ``(N,)`` bool: samples it rejects get no contact check, the
    costly stage, and score 0. ``progress`` shows a bar per stage. ``models`` are loaded
    here where not given (and let go after), for one video.

    Returns:
        The triage, and per sample that got the contact check its crop (x0, y0, x1, y1 in
        sample pixels, possibly outside the frame) and Sapiens2's ``parts_size`` labels of it.
    """
    own = models is None
    models = TriageModels(settings, device) if own else models
    frames, count = samples.frames, len(samples.frames)
    height, width = frames[0].shape[:2]

    # Stage 1: people and joints (YOLO-pose), every sample.
    pose_model = models.pose
    keypoints = np.full((count, 17, 3), np.nan)
    boxes = np.full((count, 5), np.nan)
    people = np.zeros(count, int)
    overlap = np.zeros(count)
    for start in tqdm(
        range(0, count, settings.batch), desc="YOLO-pose", unit="batch", disable=not progress,
        leave=False,
    ):  # fmt: skip
        results = pose_model.predict(
            frames[start : start + settings.batch],
            imgsz=settings.analysis_side,
            quantize=16,
            verbose=False,
            device=0 if device == "cuda" else device,
        )
        for i, result in enumerate(results, start=start):
            found = result.boxes.data.cpu().numpy()  # x0, y0, x1, y1, score, class
            joints = result.keypoints.data.cpu().numpy()  # (people, 17, 3)
            keep = found[:, 4] >= settings.min_person
            found, joints = found[keep], joints[keep]
            people[i] = len(found)
            if not len(found):
                continue
            areas = (found[:, 2] - found[:, 0]) * (found[:, 3] - found[:, 1])
            main = int(areas.argmax())
            boxes[i], keypoints[i] = found[main, :5], joints[main]
            x0, y0, x1, y1 = found[main, :4]
            for other in np.delete(found, main, axis=0):
                w = max(0, min(x1, other[2]) - max(x0, other[0]))
                h = max(0, min(y1, other[3]) - max(y0, other[1]))
                overlap[i] = max(overlap[i], w * h / areas[main])

    sharpness = np.full(count, np.nan)
    for i in np.flatnonzero(people > 0):
        sharpness[i] = focus(frames[i], boxes[i, :4])

    person_height = (boxes[:, 3] - boxes[:, 1]) / height
    joints_seen = (np.nan_to_num(keypoints[..., 2]) >= settings.joint_seen).sum(-1)
    edge = settings.edge
    at_edge = (
        (boxes[:, 0] <= edge * width)
        | (boxes[:, 1] <= edge * height)
        | (boxes[:, 2] >= (1 - edge) * width)
        | (boxes[:, 3] >= (1 - edge) * height)
    )
    median = np.nanmedian(sharpness) if np.isfinite(sharpness).any() else 1.0
    survivors = (
        (people > 0)
        & (np.nan_to_num(person_height) >= settings.min_height)
        & (joints_seen >= settings.min_joints)
        & (np.nan_to_num(sharpness) >= settings.min_sharpness * median)
    )
    if prefilter is not None:
        survivors &= prefilter(boxes, keypoints)

    # Stage 2: contact (Sapiens2 at parts_size), survivors only.
    contact, crops = _contact(frames, boxes, np.flatnonzero(survivors), settings, models, progress)
    if own:
        del models
        torch.cuda.empty_cache()

    factors = np.zeros((count, len(FACTORS)))
    factors[:, 0] = np.clip(
        (np.nan_to_num(person_height) - settings.min_height)
        / (settings.good_height - settings.min_height),
        0,
        1,
    )
    factors[:, 1] = joints_seen / 17
    factors[:, 2] = np.where(at_edge & (joints_seen < 17), 0.5, 1.0)
    factors[:, 3] = np.clip(np.nan_to_num(sharpness / median), 0, 1)
    touching = np.maximum(np.nan_to_num(contact), overlap)
    factors[:, 4] = np.clip(1 - touching / settings.max_contact, 0, 1)
    result = Triage(
        times=samples.times,
        fps=samples.fps,
        duration=samples.duration,
        sample_size=np.array([width, height]),
        shot=shot_numbers(frames, settings.shot_cut),
        people=people,
        boxes=boxes,
        keypoints=keypoints,
        overlap=overlap,
        sharpness=sharpness,
        contact=contact,
        factors=factors,
        score=np.where(survivors, factors.prod(1), 0.0),
    )
    return result, crops


def context_crop(box: np.ndarray, context: float) -> list[int]:
    """x0, y0, x1, y1 of a 3:4 crop ``context`` times a box, possibly outside the frame."""
    x0, y0, x1, y1 = box[:4]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    h = context * max(y1 - y0, (x1 - x0) * 4 / 3)
    w = h * 3 / 4
    return [round(cx - w / 2), round(cy - h / 2), round(cx + w / 2), round(cy + h / 2)]


def padded_crop(image: np.ndarray, crop: list[int], fill=GREY) -> np.ndarray:
    """The crop's pixels, ``fill`` where it leaves the image."""
    x0, y0, x1, y1 = crop
    height, width = image.shape[:2]
    out = np.full((y1 - y0, x1 - x0, 3), fill, np.uint8)
    fx0, fy0, fx1, fy1 = max(x0, 0), max(y0, 0), min(x1, width), min(y1, height)
    if fx1 > fx0 and fy1 > fy0:
        out[fy0 - y0 : fy1 - y0, fx0 - x0 : fx1 - x0] = image[fy0:fy1, fx0:fx1]
    return out


def _contact(frames, boxes, todo, settings, models, progress=False):
    """Other body pixels per body pixel of the main person, in a crop ``context`` times their
    box: body pixels outside their (slightly widened) box belong to someone else."""
    contact = np.full(len(frames), np.nan)
    crops = {}
    if not len(todo):
        return contact, crops
    processor, model = models.parts
    device = models.device
    size_h, size_w = settings.parts_size
    with torch.inference_mode():
        for start in tqdm(
            range(0, len(todo), settings.batch), desc="contact (Sapiens2)", unit="batch",
            disable=not progress, leave=False,
        ):  # fmt: skip
            batch = todo[start : start + settings.batch]
            boxes_ = [context_crop(boxes[i], settings.context) for i in batch]
            inputs = processor(
                images=[padded_crop(frames[i], c) for i, c in zip(batch, boxes_, strict=True)],
                size={"height": size_h, "width": size_w},
                return_tensors="pt",
            )
            labels = model(**inputs.to(device, torch.bfloat16)).logits.argmax(1).cpu().numpy()
            for i, crop, label in zip(batch, boxes_, labels, strict=True):
                body = (label > 0) & ~np.isin(label, CLOTHING)
                x0, y0, x1, y1 = crop
                sx, sy = size_w / (x1 - x0), size_h / (y1 - y0)
                bx0, by0, bx1, by1 = boxes[i, :4]
                margin = 0.05 * max(bx1 - bx0, by1 - by0)
                inside = np.zeros_like(body)
                inside[
                    max(0, int((by0 - margin - y0) * sy)) : int((by1 + margin - y0) * sy),
                    max(0, int((bx0 - margin - x0) * sx)) : int((bx1 + margin - x0) * sx),
                ] = True
                contact[i] = (body & ~inside).sum() / max((body & inside).sum(), 1)
                crops[int(i)] = (crop, label)
    return contact, crops


def rank_shots(times: np.ndarray, shot: np.ndarray, score: np.ndarray, end: float) -> list[Shot]:
    """Group triage samples into their shots, the best shot first.

    ``times`` (N,) are the samples' times in seconds, ascending; ``shot`` (N,) each sample's
    shot number (non-decreasing from 0); ``score`` (N,) their scores; ``end`` the video's length
    in seconds.
    """
    shots = []
    for s in np.unique(shot):
        members = np.flatnonzero(shot == s)
        after = members[-1] + 1
        shots.append(
            Shot(
                score=float(np.median(score[members])),
                begin=float(times[members[0]]),
                end=float(times[after]) if after < len(times) else float(end),
                samples=members,
                best=int(members[np.argmax(score[members])]),
            )
        )
    return sorted(shots, key=lambda s: -s.score)


def good_stretches(
    times: np.ndarray,
    shot: np.ndarray,
    score: np.ndarray,
    end: float,
    min_score: float,
    max_dip: int,
) -> list[Shot]:
    """The stretches of each shot whose samples score at least ``min_score``, best first.

    A long shot is mostly not good all through: whoever is filmed walks out of it, turns away
    or is hidden now and then, and a shot's median drags its good minutes down with its bad
    ones. A stretch bridges at most ``max_dip`` worse samples in a row (a passing arm); it never
    crosses a shot cut, where motion is not smooth. It begins at its first good sample and ends
    at the sample after its last (or the shot's end); its score is its samples' median.

    Args: as ``rank_shots``.
    """
    stretches = []
    for s in np.unique(shot):
        members = np.flatnonzero(shot == s)
        good = score[members] >= min_score
        runs, current, dip = [], [], 0
        for index, is_good in zip(members, good, strict=True):
            if is_good:
                current, dip = current + [index], 0
            elif current and dip < max_dip:
                current, dip = current + [index], dip + 1
            else:
                if current:
                    runs.append(current[: len(current) - dip])
                current, dip = [], 0
        if current:
            runs.append(current[: len(current) - dip])
        last = members[-1]
        for run in runs:
            run = np.array(run)
            after = run[-1] + 1
            stretches.append(
                Shot(
                    score=float(np.median(score[run])),
                    begin=float(times[run[0]]),
                    end=float(times[after]) if run[-1] < last else _shot_end(times, last, end),
                    samples=run,
                    best=int(run[np.argmax(score[run])]),
                )
            )
    return sorted(stretches, key=lambda s: -s.score)


def _shot_end(times: np.ndarray, last: int, end: float) -> float:
    """The time a shot whose last sample is ``last`` ends: the next sample's, or the video's
    end."""
    return float(times[last + 1]) if last + 1 < len(times) else float(end)


@dataclass(frozen=True)
class Piece:
    """A stretch of a shot short enough to prepare at once."""

    shot: int
    """The shot's rank among the chosen ones, best first."""
    score: float
    """The shot's score."""
    first: int
    """The video frame it starts at."""
    count: int
    """Frames, every ``frame_step``-th from ``first``."""
    select: int
    """The frame (of its ``count``) the person is picked on: its best triage sample's, or its
    middle."""


def shot_pieces(
    shots: list[Shot],
    times: np.ndarray,
    score: np.ndarray,
    fps: float,
    frame_step: int,
    max_frames: int,
    min_frames: int,
) -> list[Piece]:
    """Cut shots into pieces of at most ``max_frames`` frames (every ``frame_step``-th); the
    shot's tail is dropped when shorter than ``min_frames``."""
    pieces = []
    for rank, shot in enumerate(shots):
        begin, end = round(shot.begin * fps), round(shot.end * fps)
        span = max_frames * frame_step
        for first in range(begin, end, span):
            count = min(max_frames, -(-(min(first + span, end) - first) // frame_step))
            if count < min_frames:
                continue
            inside = [
                i
                for i in shot.samples
                if first <= round(times[i] * fps) < first + count * frame_step
            ]
            if inside:
                best = max(inside, key=lambda i: score[i])
                select = min((round(times[best] * fps) - first) // frame_step, count - 1)
            else:
                select = count // 2
            pieces.append(Piece(rank, shot.score, first, count, int(select)))
    return pieces
