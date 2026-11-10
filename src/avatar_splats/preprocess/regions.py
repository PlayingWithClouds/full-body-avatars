"""Close-ups of her breasts and vulva: found, measured and scored the way ``quality`` does faces,
to refine those regions of the avatar from the many frames that show them up close.

The detector is Pakaho's yolo26s NSFW detector, v3 (``Pakaho-q1/yolo26-nsfw-labels`` on Hugging
Face): boxes of 19 classes, among them exposed breasts and genitalia, and hands. Its checkpoint is a
pickle from an anonymous uploader, so it is never loaded here: it was opened once in a sandbox
(bubblewrap, no network, read-only) and its weights saved as safetensors, its architecture as
YAML and its class names as JSON (``yolo_model`` reads the three). NSFW-API's breast
segmentation (``NSFW-API/NSFW_Segmentation``, ``nsfw-seg-breast-x``: breast, areola and nipple
masks) was converted the same way, for the fit to come. Its vulva model finds nothing on our
footage, and no model gives keypoints there.

A region is measured where the detector finds it inside her box:

- resolution: its box's longer side in the source video's pixels (both breasts: the box around
  both),
- sharpness and light: as a face's, over the ellipses inside its boxes,
- occlusion: the share of those ellipses the detector's hand boxes cover,
- found: found at all, and not cut off by the frame's edge,
- bare: how little of those ellipses Sapiens2's body parts call clothing, where they were
  segmented (``clothed_share``). The detector takes tight, skin-coloured satin over the crotch for
  bare genitalia with 0.77 confidence; the segmentation runs on the few frames that could pass,
  not on all.

Identity is the frame's own where a face is recognised, else its shot's (``shot_identity``): a
close-up rarely shows her face, but the shot it is in mostly did, a moment before or after.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm.auto import tqdm

from avatar_splats.preprocess.quality import _identity, _ramp, _verdicts, at_edge, lighting
from avatar_splats.preprocess.triage import focus

DETECTOR = "pakaho-nsfw-v3"
REGIONS = {"breasts": "FEMALE_BREAST_EXPOSED", "vulva": "FEMALE_GENITALIA_EXPOSED"}
"""Per region the detector's class of it."""
PARTS = {"breasts": 2, "vulva": 1}
"""Per region how many boxes of its class it is made of at most."""
HAND = "HAND"
REGION_FACTOR_NAMES = (
    "found", "resolution", "sharpness", "brightness", "evenness", "occlusion", "bare", "alone",
    "identity",
)  # fmt: skip
CLOTHING = ("Apparel", "Upper_Clothing", "Lower_Clothing")
"""Sapiens2's labels (``prepare.PART_NAMES``) of clothes."""
LIGHT_BLUR = 0.05
"""The region's size: the Gaussian evenness is measured after (``quality.lighting``)."""
DETECTION_COLUMNS = 7
"""A stored detection: the frame's index, x0, y0, x1, y1, confidence, class."""


def yolo_model(directory: Path, name: str, task: str, device: str):
    """An ultralytics YOLO model from ``name``'s safetensors weights, YAML architecture and JSON
    class names in ``directory``; no pickle is loaded. ``task``: "detect" or "segment"."""
    import torch
    import yaml
    from safetensors.torch import load_file
    from ultralytics import YOLO
    from ultralytics.cfg import DEFAULT_CFG_DICT
    from ultralytics.nn.tasks import DetectionModel, SegmentationModel
    from ultralytics.utils import LOGGER

    architecture = yaml.safe_load((directory / f"{name}.yaml").read_text())
    names = {int(k): v for k, v in json.loads((directory / f"{name}.json").read_text()).items()}
    # YOLO(path.yaml) takes the model's size from the file's name, not from the file: built here
    network = {"detect": DetectionModel, "segment": SegmentationModel}[task](
        architecture, verbose=False
    )
    network.load_state_dict(load_file(directory / f"{name}.safetensors"))
    network.names = names
    level = LOGGER.level
    LOGGER.setLevel("ERROR")  # it builds a smallest-size model first, and says so
    try:
        model = YOLO(directory / f"{name}.yaml", task=task)
    finally:
        LOGGER.setLevel(level)
    model.model = network.to(device).eval()
    network.task = task
    network.args = {**DEFAULT_CFG_DICT, **model.overrides}
    torch.cuda.empty_cache()
    return model


class RegionDetector:
    """The NSFW detector (see the module's docstring), on batches of frames."""

    def __init__(
        self, directory: Path, device: str, batch: int = 32, min_confidence: float = 0.25,
        image_size: int = 640,
    ):  # fmt: skip
        self.model = yolo_model(directory, DETECTOR, "detect", device)
        self.names = self.model.model.names
        self.device, self.batch = device, batch
        self.min_confidence, self.image_size = min_confidence, image_size

    def __call__(self, frames, progress: bool = False) -> list[np.ndarray]:
        """Per RGB uint8 frame its ``(K, 6)`` detections: x0, y0, x1, y1 in its pixels,
        confidence, class (an index into ``names``)."""
        found = []
        starts = range(0, len(frames), self.batch)
        for start in tqdm(starts, desc="regions", unit="batch", disable=not progress, leave=False):
            chunk = [np.ascontiguousarray(f[..., ::-1]) for f in frames[start : start + self.batch]]
            results = self.model.predict(
                chunk, device=self.device, conf=self.min_confidence, imgsz=self.image_size,
                quantize=32 if self.device == "cpu" else 16, verbose=False,
            )  # fmt: skip
            for result in results:
                boxes = result.boxes
                found.append(
                    np.column_stack(
                        [
                            boxes.xyxy.cpu().numpy(),
                            boxes.conf.cpu().numpy(),
                            boxes.cls.cpu().numpy(),
                        ]
                    ).astype(np.float32)  # fmt: skip
                    if len(boxes)
                    else np.zeros((0, 6), np.float32)
                )
        return found


def detections_table(found: list[np.ndarray]) -> np.ndarray:
    """``(M, 7)`` all frames' detections in one array, each with its frame's index first."""
    rows = [np.column_stack([np.full(len(f), i), f]) for i, f in enumerate(found) if len(f)]
    return np.concatenate(rows) if rows else np.zeros((0, DETECTION_COLUMNS), np.float32)


def detections_of(table: np.ndarray, count: int) -> list[np.ndarray]:
    """``detections_table``'s inverse: per frame of ``count`` its ``(K, 6)`` detections."""
    frames = table[:, 0].astype(int)
    return [table[frames == i, 1:] for i in range(count)]


def class_index(names: dict[int, str], name: str) -> int:
    return next(k for k, v in names.items() if v == name)


def region_parts(
    detections: np.ndarray, names: dict[int, str], region: str, person: np.ndarray | None
) -> np.ndarray:
    """``(P, 5)`` the boxes (x0, y0, x1, y1, confidence) of a region in one frame's
    ``(K, 6)`` detections: its class's, centred inside the ``person`` box where that is given
    and finite (hers, not someone else's), the ``PARTS[region]`` most confident."""
    mine = detections[detections[:, 5] == class_index(names, REGIONS[region])]
    if person is not None and np.isfinite(person[:4]).all() and len(mine):
        cx, cy = (mine[:, 0] + mine[:, 2]) / 2, (mine[:, 1] + mine[:, 3]) / 2
        inside = (cx >= person[0]) & (cx <= person[2]) & (cy >= person[1]) & (cy <= person[3])
        mine = mine[inside]
    return mine[np.argsort(-mine[:, 4])][: PARTS[region], :5]


def parts_mask(parts: np.ndarray, shape: tuple[int, int], shift=(0, 0)) -> np.ndarray:
    """``(H, W)`` bool: the ellipses inscribed in ``(P, 4+)`` boxes, moved by ``-shift``."""
    mask = np.zeros(shape, np.uint8)
    for x0, y0, x1, y1 in parts[:, :4] - np.array([*shift, *shift]):
        centre = (round((x0 + x1) / 2), round((y0 + y1) / 2))
        axes = (max(1, round((x1 - x0) / 2)), max(1, round((y1 - y0) / 2)))
        cv2.ellipse(mask, centre, axes, 0, 0, 360, 1, -1)
    return mask.astype(bool)


def clothed_share(labels: np.ndarray, parts: np.ndarray) -> float:
    """The share of the ellipses in ``(P, 4+)`` boxes that ``(H, W)`` Sapiens2 labels
    (``prepare.PART_NAMES``) call clothing, of those labelled as the person; NaN where none is."""
    from avatar_splats.preprocess.prepare import NOT_THEM, PART_NAMES

    mask = parts_mask(parts, labels.shape)
    person = mask & (labels != NOT_THEM) & (labels != PART_NAMES.index("Background"))
    if not person.any():
        return float("nan")
    clothing = np.isin(labels, [PART_NAMES.index(name) for name in CLOTHING])
    return float((clothing & person).sum() / person.sum())


def worth_segmenting(
    detections: list[np.ndarray],
    names: dict[int, str],
    persons: np.ndarray | None,
    source_scale: float,
    settings: dict[str, RegionQualitySettings],
) -> np.ndarray:
    """``(N,)`` bool: the frames with a region big enough to pass its resolution gate, the only
    ones ``clothed_share`` is worth Sapiens2's time on."""
    out = np.zeros(len(detections), bool)
    for i, found in enumerate(detections):
        for region, s in settings.items():
            parts = region_parts(found, names, region, None if persons is None else persons[i])
            if len(parts):
                size = max(
                    parts[:, 2].max() - parts[:, 0].min(), parts[:, 3].max() - parts[:, 1].min()
                )
                out[i] |= size / source_scale >= s.gates.get("resolution", 0.0) * s.good_pixels
    return out


def shot_identity(identity: np.ndarray, shot: np.ndarray) -> np.ndarray:
    """``(N,)`` each sample's similarity to the actor where its face was recognised, else the
    median of its shot's recognised faces; NaN where its shot shows none."""
    out = np.asarray(identity, float).copy()
    for value in np.unique(shot):
        inside = shot == value
        known = out[inside][np.isfinite(out[inside])]
        if len(known):
            out[inside & ~np.isfinite(out)] = np.median(known)
    return out


@dataclass(frozen=True)
class RegionQualitySettings:
    """``quality.FaceQualitySettings``' counterparts for a region: how good a close-up is."""

    edge: float
    good_pixels: float
    """The region's box's longer side in the source's pixels that earns full resolution."""
    good_focus: float
    dark: float
    bright: float
    max_clipped: float
    uneven: float
    even: float
    max_occlusion: float
    """Share of the region the hands' boxes cover at which occlusion is 0."""
    max_clothed: float
    """Share of the region labelled clothing at which bare is 0."""
    max_overlap: float
    other_face: float
    same_face: float
    weights: dict[str, float]
    gates: dict[str, float]
    min_score: float


def measure_regions(
    frames,
    detections: list[np.ndarray],
    names: dict[int, str],
    region: str,
    persons: np.ndarray | None,
    source_scale: float,
    overlap: np.ndarray,
    settings: RegionQualitySettings,
    labels: list[np.ndarray | None] | None = None,
    progress: bool = False,
) -> dict[str, np.ndarray]:
    """Raw measures of N frames' region: ``box`` ``(N, 4)`` around its parts, and ``(N,)``
    each of ``confidence`` (its parts' mean), ``parts``, ``pixels`` (its box's longer side in
    source pixels), ``focus``, ``brightness``, ``clipped``, ``evenness``, ``occlusion``,
    ``clothed`` (``clothed_share``), ``found``, ``cut`` and ``overlap``; NaN where it was not
    found or, ``clothed``, not segmented.

    Args:
        frames: N ``(H, W, 3)`` RGB uint8 frames.
        detections: per frame ``RegionDetector``'s ``(K, 6)``.
        persons: ``(N, 4+)`` her box per frame (NaN: not known, any region counts), or None.
        source_scale: the frames' size over the source video's.
        overlap: ``(N,)`` the largest share of her box another person's covers.
        labels: per frame its ``(H, W)`` Sapiens2 body parts, or None where not segmented.
    """
    count = len(frames)
    names_ = ("confidence", "parts", "pixels", "focus", "brightness", "clipped", "evenness",
              "occlusion", "clothed")  # fmt: skip
    out = {name: np.full(count, np.nan) for name in names_}
    out["box"] = np.full((count, 4), np.nan)
    out["found"] = np.zeros(count, bool)
    out["cut"] = np.zeros(count, bool)
    out["overlap"] = np.asarray(overlap, float)
    hand = class_index(names, HAND)
    for i in tqdm(range(count), desc=region, unit="frame", disable=not progress, leave=False):
        parts = region_parts(detections[i], names, region, None if persons is None else persons[i])
        if not len(parts):
            continue
        shape = frames[i].shape[:2]
        low, high = parts[:, :2].min(0), parts[:, 2:4].max(0)
        x0, y0 = max(int(low[0]), 0), max(int(low[1]), 0)
        x1, y1 = min(int(np.ceil(high[0])), shape[1]), min(int(np.ceil(high[1])), shape[0])
        if x1 <= x0 or y1 <= y0:
            continue
        crop = frames[i][y0:y1, x0:x1]
        mask = parts_mask(parts, crop.shape[:2], (x0, y0))
        if not mask.any():
            continue
        hands = detections[i][detections[i][:, 5] == hand]
        covered = np.zeros(crop.shape[:2], bool)
        for hx0, hy0, hx1, hy1 in hands[:, :4] - np.array([x0, y0, x0, y0]):
            covered[max(int(hy0), 0) : max(int(np.ceil(hy1)), 0),
                    max(int(hx0), 0) : max(int(np.ceil(hx1)), 0)] = True  # fmt: skip
        size = float(max(high - low))
        out["box"][i] = (*low, *high)
        out["confidence"][i] = parts[:, 4].mean()
        out["parts"][i] = len(parts)
        out["found"][i] = True
        out["pixels"][i] = size / source_scale
        out["focus"][i] = focus(crop, np.array([0.0, 0.0, x1 - x0, y1 - y0]), mask)
        brightness, clipped, evenness = lighting(crop, mask, LIGHT_BLUR * size)
        out["brightness"][i], out["clipped"][i], out["evenness"][i] = brightness, clipped, evenness
        out["occlusion"][i] = float((covered & mask).sum() / mask.sum())
        out["cut"][i] = at_edge(np.array([*low, *high]), shape, settings.edge)
        if labels is not None and labels[i] is not None:
            out["clothed"][i] = clothed_share(labels[i], parts)
    return out


def score_regions(
    measures: dict[str, np.ndarray], settings: RegionQualitySettings
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, list[str]]:
    """``quality.score_faces`` for a region: the ``REGION_FACTOR_NAMES``, the score, whether
    each frame passes and why not."""
    s = settings
    factors = {
        "found": np.where(measures["found"], np.where(measures["cut"], 0.5, 1.0), 0.0),
        "resolution": _ramp(measures["pixels"], 0, s.good_pixels),
        "sharpness": _ramp(measures["focus"], 0, s.good_focus),
        "brightness": _ramp(measures["brightness"], s.dark, s.bright)
        * _ramp(measures["clipped"], s.max_clipped, 0),
        "evenness": _ramp(measures["evenness"], s.uneven, s.even),
        "occlusion": _ramp(measures["occlusion"], s.max_occlusion, 0),
        # 1 where not segmented: triage's samples, and frames too small to pass anyway
        "bare": np.where(
            np.isfinite(measures["clothed"]), _ramp(measures["clothed"], s.max_clothed, 0), 1.0
        ),
        "alone": _ramp(np.nan_to_num(measures["overlap"], nan=1.0), s.max_overlap, 0),
        "identity": _identity(measures, s),
    }
    measured_by = {
        "resolution": "pixels", "sharpness": "focus", "brightness": "brightness",
        "evenness": "evenness", "occlusion": "occlusion",
    }  # fmt: skip
    return (factors, *_verdicts(factors, measures, measured_by, s))
