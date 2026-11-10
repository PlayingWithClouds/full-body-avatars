"""Close-ups of the breasts as the fit sees them: each harvested frame (notebook 14) decoded at
the source's resolution, cropped square around the breasts and scaled to ``CROP_SIZE``, with what
the models see in the crop:

- NSFW-API's breast segmentation (``nsfw-seg-breast-x``, converted like the detector, see
  ``preprocess.regions``): the breasts' and areolae's masks and the nipples' centres,
- Sapiens2's body parts (``prepare.BodyParts``): her torso's outline, and what hides it,
- Sapiens2's surface normals, in the camera's axes,
- ViTPose's keypoints, from the prepared piece.

Nipples are not told apart: a crop shows one or two, and seen from the front her right one is on
the image's left, from behind not at all; the fit pairs each with the nearer of the model's.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from avatar_splats.data.video import read_frames
from avatar_splats.preprocess.regions import detections_of, region_parts, yolo_model
from avatar_splats.records import ArrayRecord

CROP_SIZE = 384
"""Pixels along a close-up's side."""
CROP_SPAN = 2.2
"""A close-up's side over the larger side of the box around the breasts: the torso's outline
around them is in it."""
BREAST_SEGMENTER = "nsfw-api-breast-x"
NSFW_CLASSES = {"breast": 0, "areola": 1, "nipple": 2}
MIN_SEGMENT = 0.3
"""NSFW-API's confidence to take a mask."""
SAME_NIPPLE = 0.04
"""Share of the crop's side within which two nipple masks are one nipple found twice."""


@dataclass(frozen=True)
class Closeups(ArrayRecord):
    """N close-ups of her breasts, each a ``CROP_SIZE`` square."""

    keys: np.ndarray
    """``(N,)`` str: "video:frame"."""
    videos: np.ndarray
    """``(N,)`` str: the harvest's video names."""
    frames: np.ndarray
    """``(N,)`` the frame indices in their videos."""
    images: np.ndarray
    """``(N, S, S, 3)`` uint8 RGB."""
    K: np.ndarray
    """``(N, 3, 3)`` intrinsics in the crop's pixels (the source's field of view)."""
    crops: np.ndarray
    """``(N, 4)`` x0, y0, x1, y1 of the crop in the source frame's pixels."""
    labels: np.ndarray
    """``(N, S, S)`` uint8 Sapiens2 body parts (``prepare.PART_NAMES``, ``NOT_THEM`` outside)."""
    normals: np.ndarray
    """``(N, S, S, 3)`` float16 Sapiens2's unit normals in the camera's axes (x right, y down,
    z forward); zero outside her."""
    breasts: np.ndarray
    """``(N, S, S)`` bool: NSFW-API's breast masks."""
    areolae: np.ndarray
    """``(N, S, S)`` bool: its areola masks."""
    nipples: np.ndarray
    """``(N, 2, 2)`` x, y in the crop's pixels of up to two nipples (or, where a nipple is not
    found, its areola's centre), in no order; NaN where fewer."""
    keypoints: np.ndarray
    """``(N, 17, 3)`` ViTPose's COCO keypoints in the crop's pixels, and confidence."""
    identity: np.ndarray
    """``(N,)`` her likeness to the actor (the piece's faces, else its shot's); NaN unknown."""


def intrinsics(fov_y_degrees: float, width: int, height: int) -> np.ndarray:
    """``(3, 3)`` a centred pinhole camera's intrinsics for a frame of ``width`` by ``height``."""
    focal = 0.5 * height / math.tan(math.radians(fov_y_degrees) / 2)
    return np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])


def crop_intrinsics(K: np.ndarray, crop: np.ndarray, size: int) -> np.ndarray:
    """``(3, 3)`` ``K`` (of the frame) for its ``crop`` (x0, y0, x1, y1) scaled to ``size``."""
    scale = size / (crop[2] - crop[0])
    out = K.copy()
    out[0, 2] -= crop[0]
    out[1, 2] -= crop[1]
    out[:2] *= scale
    return out


def frame_intrinsics(closeups: Closeups, i: int) -> np.ndarray:
    """``(3, 3)`` the source frame's intrinsics of close-up ``i``, from its crop's."""
    crop, K = closeups.crops[i], closeups.K[i].astype(np.float64)
    scale = (crop[2] - crop[0]) / closeups.images.shape[1]
    out = K.copy()
    out[:2] *= scale
    out[0, 2] += crop[0]
    out[1, 2] += crop[1]
    return out


def square_crop(box: np.ndarray, span: float) -> np.ndarray:
    """``(4,)`` a square ``span`` times ``box``'s larger side, around its centre."""
    centre = (box[:2] + box[2:4]) / 2
    half = span * max(box[2] - box[0], box[3] - box[1]) / 2
    return np.array([centre[0] - half, centre[1] - half, centre[0] + half, centre[1] + half])


def cut(frame: np.ndarray, crop: np.ndarray, size: int) -> np.ndarray:
    """``(size, size, 3)`` the ``crop`` of an RGB frame, black where it reaches past the frame."""
    x0, y0 = crop[:2]
    scale = size / (crop[2] - crop[0])
    matrix = np.array([[scale, 0, -x0 * scale], [0, scale, -y0 * scale]])
    return cv2.warpAffine(frame, matrix, (size, size), flags=cv2.INTER_AREA, borderValue=0)


def nipple_centres(nipples: list, areolae: list) -> np.ndarray:
    """``(2, 2)`` up to two nipple centres from ``(H, W)`` bool masks: the nipples', and the
    centres of areolae with no nipple in them; NaN where fewer."""
    out = np.full((2, 2), np.nan)
    centres = []
    for mask in nipples:
        ys, xs = np.nonzero(mask)
        near = SAME_NIPPLE * mask.shape[1]
        if len(xs) and all(np.hypot(xs.mean() - x, ys.mean() - y) > near for x, y in centres):
            centres.append((xs.mean(), ys.mean()))
    for mask in areolae:  # an areola counts where no nipple was found within its size of it
        ys, xs = np.nonzero(mask)
        reach = 1.5 * np.sqrt(len(xs) / np.pi)
        if len(xs) and all(np.hypot(xs.mean() - x, ys.mean() - y) > reach for x, y in centres):
            centres.append((xs.mean(), ys.mean()))
    for k, centre in enumerate(centres[:2]):
        out[k] = centre
    return out


def segment_breasts(model, images: np.ndarray, batch: int = 8):
    """Per ``(S, S, 3)`` RGB image NSFW-API's ``(S, S)`` breast and areola masks and its
    ``(2, 2)`` nipple centres (``nipple_centres``)."""
    out = []
    for start in range(0, len(images), batch):
        chunk = [np.ascontiguousarray(i[..., ::-1]) for i in images[start : start + batch]]
        results = model.predict(
            chunk, conf=MIN_SEGMENT, retina_masks=True, verbose=False, device=model.device
        )
        for image, result in zip(chunk, results, strict=True):
            size = image.shape[:2]
            found = {name: [] for name in NSFW_CLASSES}
            if result.masks is not None:
                masks = result.masks.data.cpu().numpy() > 0.5
                for mask, cls in zip(masks, result.boxes.cls.cpu().numpy(), strict=True):
                    name = next(n for n, c in NSFW_CLASSES.items() if c == int(cls))
                    found[name].append(cv2.resize(mask.astype(np.uint8), size[::-1]) > 0)
            union = {
                name: np.any(found[name], axis=0) if found[name] else np.zeros(size, bool)
                for name in ("breast", "areola")
            }
            out.append(
                (union["breast"], union["areola"], nipple_centres(found["nipple"], found["areola"]))
            )
    return out


@torch.inference_mode()
def estimate_normals(
    images: np.ndarray, model_id: str, device: str, batch: int = 8, size: int = 768
) -> np.ndarray:
    """``(N, S, S, 3)`` float16 Sapiens2's unit normals of ``(N, S, S, 3)`` RGB crops, in the
    camera's axes (``surface.TO_CAMERA_AXES``)."""
    from transformers import AutoImageProcessor, Sapiens2ForNormalEstimation

    from avatar_splats.preprocess.surface import TO_CAMERA_AXES

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = Sapiens2ForNormalEstimation.from_pretrained(model_id, dtype=torch.bfloat16)
    model = model.to(device).eval()
    out = np.zeros((*images.shape[:3], 3), np.float16)
    axes = torch.tensor(TO_CAMERA_AXES, device=device)[:, None, None]
    for start in range(0, len(images), batch):
        chunk = [torch.from_numpy(i).to(device) for i in images[start : start + batch]]
        inputs = processor(
            images=chunk, return_tensors="pt", device=device,
            size={"height": size, "width": size},
        )  # fmt: skip
        normals = model(**inputs.to(device, torch.bfloat16)).normals.float() * axes
        normals = F.interpolate(normals, images.shape[1:3], mode="bilinear", align_corners=False)
        out[start : start + len(chunk)] = (
            F.normalize(normals, dim=1).permute(0, 2, 3, 1).half().cpu().numpy()
        )
    del model
    torch.cuda.empty_cache()
    return out


def gather_closeups(
    records: list[dict],
    video_paths: dict[str, Path],
    fov_y_degrees: dict[str, float],
    region_names: dict[int, str],
    models: Path,
    parts_model: str,
    normal_model: str,
    device: str,
    progress: bool = True,
) -> Closeups:
    """The close-ups of harvest records (rows of notebook 14's ``harvest.csv`` of stage
    "prepared": ``video``, ``frame``, ``piece``, ``row``, ``identity``, ``shot_identity``).

    Args:
        video_paths: per harvest video name its file.
        fov_y_degrees: per video name its vertical field of view; 35 where not given.
        region_names: the NSFW detector's classes.
        models: the folder of the converted NSFW models.
        parts_model, normal_model: Sapiens2's body part and normal models.
    """
    from avatar_splats.preprocess.prepare import BodyParts

    rows = []
    for record in tqdm(records, desc="close-ups", unit="frame", disable=not progress):
        piece = np.load(record["piece"])
        row = int(record["row"])
        scale = float(piece["scale"])
        stored = Path(record["piece"]).with_name(Path(record["piece"]).stem + "-regions.npz")
        detections = detections_of(np.load(stored)["detections"], len(piece["frame_indices"]))
        parts = region_parts(detections[row], region_names, "breasts", piece["boxes"][row])
        if not len(parts):
            continue
        box = np.array([*parts[:, :2].min(0), *parts[:, 2:4].max(0)]) / scale
        path = video_paths[record["video"]]
        frame = next(read_frames(path, max_frames=1, start=int(record["frame"])))
        crop = square_crop(box, CROP_SPAN)
        height, width = frame.shape[:2]
        fov = fov_y_degrees.get(record["video"], 35.0)
        keypoints = piece["keypoints"][row].copy()
        keypoints[:, :2] = (keypoints[:, :2] / scale - crop[:2]) * CROP_SIZE / (crop[2] - crop[0])
        identity = float(record["identity"])
        if not np.isfinite(identity):
            identity = float(record.get("shot_identity", "nan") or "nan")
        rows.append({
            "key": f"{record['video']}:{record['frame']}", "video": record["video"],
            "frame": int(record["frame"]), "image": cut(frame, crop, CROP_SIZE),
            "K": crop_intrinsics(intrinsics(fov, width, height), crop, CROP_SIZE),
            "crop": crop, "keypoints": keypoints, "identity": identity,
        })  # fmt: skip
    images = np.stack([r["image"] for r in rows])
    segmenter = yolo_model(models, BREAST_SEGMENTER, "segment", device)
    segmented = segment_breasts(segmenter, images)
    del segmenter
    whole = np.array([[0.0, 0.0, CROP_SIZE, CROP_SIZE]] * len(images))
    labels, _ = BodyParts(parts_model, 8, 0.0, device)(list(images), whole)
    torch.cuda.empty_cache()
    normals = estimate_normals(images, normal_model, device)
    return Closeups(
        keys=np.array([r["key"] for r in rows]),
        videos=np.array([r["video"] for r in rows]),
        frames=np.array([r["frame"] for r in rows]),
        images=images,
        K=np.stack([r["K"] for r in rows]).astype(np.float32),
        crops=np.stack([r["crop"] for r in rows]).astype(np.float32),
        labels=np.asarray(labels, np.uint8),
        normals=normals,
        breasts=np.stack([s[0] for s in segmented]),
        areolae=np.stack([s[1] for s in segmented]),
        nipples=np.stack([s[2] for s in segmented]).astype(np.float32),
        keypoints=np.stack([r["keypoints"] for r in rows]).astype(np.float32),
        identity=np.array([r["identity"] for r in rows]),
    )
