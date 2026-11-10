"""What a frame says about the body's surface beyond its outline: which way it faces and how far
it is.

- Sapiens2's surface normals, on the same 3:4 crop around the person as its body parts,
- Sapiens2's pointmap there: a 3D point per pixel of the person, in metres by its own guess
  of the scale; its depth tells which limb is in front, and its points against their pixels
  give the focal length (``pointmap_focal``),
- Depth Pro's metric depth of the whole frame, and its estimate of the field of view.

The maps are kept at a smaller size (``map_side`` pixels along the longer side) than the
frames: they guide the fit at that size.

Sapiens2 gives normals with x right, y up and z towards the camera (OpenGL's axes): checked
against the outline, where the surface turns away from the camera and its normal points out of
the mask, and over the person, who faces the camera. Its pointmap has x right, y down and z
forward (OpenCV's). Both are turned into the cameras' axes here (OpenCV's: a surface facing
the camera has a normal with negative z).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from avatar_splats.preprocess.prepare import padded_on, person_crop
from avatar_splats.records import ArrayRecord

TO_CAMERA_AXES = (1.0, -1.0, -1.0)
"""Sapiens2's normals (OpenGL axes) times these are in the cameras' axes (OpenCV's)."""


def lift_keypoints(
    keypoints: np.ndarray,
    point_depth: np.ndarray,
    image_size: tuple[int, int],
    min_confidence: float,
    radius: int = 2,
) -> np.ndarray:
    """``(N, 17)`` the pointmap's depth of the skin at each confident COCO keypoint's pixel:
    the median over the map pixels within ``radius`` that show the person.

    That skin may be the keypoint's own or that of whatever hides it (a hip behind a thigh);
    which, Sapiens2's body parts cannot tell on bare skin (they label legs and arms as
    clothing or hands), so the fit takes these as lopsided targets (``keypoint_depth_loss``).

    Args:
        keypoints: ``(N, 17, 3)`` x, y in frame pixels and confidence.
        point_depth: ``(N, h, w)`` ``SurfaceMaps.point_depth``, NaN off the person.
        image_size: the frames' width and height.

    Returns:
        Depth along the view direction in the pointmap's units (metres by its guess of the
        scale); NaN where unsure or off the person.
    """
    width, height = image_size
    count, map_height, map_width = point_depth.shape
    out = np.full(keypoints.shape[:2], np.nan, np.float32)
    rows, columns = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    for i in range(count):
        for k in range(keypoints.shape[1]):
            x, y, confidence = keypoints[i, k]
            if not confidence >= min_confidence:
                continue
            my = np.clip(_at(y, height, map_height) + rows, 0, map_height - 1)
            mx = np.clip(_at(x, width, map_width) + columns, 0, map_width - 1)
            depths = point_depth[i, my, mx].astype(np.float32)
            if np.isfinite(depths).any():
                out[i, k] = np.nanmedian(depths)
    return out


def _at(coordinate, size: int, cells: int):
    """The index among ``cells`` cells of a coordinate on an axis ``size`` long."""
    return np.clip(np.floor(np.asarray(coordinate) / size * cells).astype(int), 0, cells - 1)


@dataclass(frozen=True)
class SurfaceSettings:
    normal_model: str
    """A Sapiens2 normal estimation model."""
    pointmap_model: str
    """A Sapiens2 pointmap estimation model."""
    depth_model: str
    """A Depth Pro model."""
    normal_size: tuple[int, int]
    """Height and width the normal model sees the crops at; its own is 1024 by 768, smaller
    is faster and needs less memory (its head decodes at this size). The pointmap model sees
    them at its own (its scale head is sized for it)."""
    batch: int
    """Person crops per batch of the Sapiens2 models."""
    depth_batch: int
    """Frames per batch of Depth Pro."""
    person_margin: float
    """Crop margin around a person's box, in shares of its larger side."""
    map_side: int
    """Pixels along the maps' longer side, at most the frames'."""


@dataclass(frozen=True)
class SurfaceMaps(ArrayRecord):
    """Surface maps of some frames, at ``map_size``; frame pixel ``p`` is map pixel
    ``p * map_size / image_size``."""

    keys: np.ndarray
    """``(N,)`` str: which frames, as the caller named them (to check reused maps)."""
    image_size: np.ndarray
    """``(2,)`` the frames' width and height."""
    map_size: np.ndarray
    """``(2,)`` the maps' width and height."""
    normals: np.ndarray
    """``(N, h, w, 3)`` float16 unit normals in the camera's axes (x right, y down, z forward;
    facing the camera is negative z); zero outside the person."""
    point_depth: np.ndarray
    """``(N, h, w)`` float16 Sapiens2's pointmap depth along the view direction, in metres by
    its guess of the scale; NaN outside the person."""
    point_focal: np.ndarray
    """``(N,)`` the focal length its points imply, in frame pixels; NaN where too few."""
    depth: np.ndarray
    """``(N, h, w)`` float16 Depth Pro's depth along the view direction, in metres, of the
    whole frame."""
    depth_focal: np.ndarray
    """``(N,)`` Depth Pro's focal length, in frame pixels."""


@torch.inference_mode()
def estimate_surfaces(
    frames: np.ndarray,
    masks: np.ndarray,
    boxes: np.ndarray,
    keys: list[str],
    settings: SurfaceSettings,
    device: str,
) -> SurfaceMaps:
    """Maps of ``(N, H, W, 3)`` RGB frames of one person, their ``(N, H, W)`` bool masks and
    ``(N, 4)`` boxes (x0, y0, x1, y1 in pixels).

    The models run once each over the frames, one after the other, each loaded for its turn
    and let go after it: the three would not fit beside each other on a smaller GPU.
    """
    from transformers import (
        AutoImageProcessor,
        DepthProForDepthEstimation,
        Sapiens2ForNormalEstimation,
        Sapiens2ForPointmapEstimation,
    )

    s = settings
    count, (height, width) = len(frames), frames.shape[1:3]
    factor = min(1.0, s.map_side / max(width, height))
    map_width, map_height = round(width * factor), round(height * factor)
    small_masks = (
        F.interpolate(
            torch.from_numpy(masks).to(device)[:, None].float(),
            (map_height, map_width),
            mode="area",
        )[:, 0]
        > 0.5
    )
    jobs = [
        (i, person_crop(boxes[i], s.person_margin))
        for i in range(count)
        if np.isfinite(boxes[i]).all()
    ]

    def crop_outputs(model_type, model_id, run):
        # (i, crop, values on the map, window, inside the person) per job, the values from
        # run(processor, model, images) -> (B, C, h, w)
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = model_type.from_pretrained(model_id, dtype=torch.bfloat16).to(device).eval()
        for start in range(0, len(jobs), s.batch):
            batch = jobs[start : start + s.batch]
            images = [padded_on(frames[i], crop, device) for i, crop in batch]
            values = run(processor, model, images).float()
            for (i, crop), value in zip(batch, values, strict=True):
                placed = _place(value, crop, factor, map_width, map_height)
                if placed is not None:
                    yield i, placed[0], placed[1], small_masks[i][placed[1]]
        del model
        torch.cuda.empty_cache()

    normal_size = {"height": s.normal_size[0], "width": s.normal_size[1]}

    def normals_of(processor, model, images):
        inputs = processor(images=images, return_tensors="pt", device=device, size=normal_size)
        out = model(**inputs.to(device, torch.bfloat16)).normals
        return out * out.new_tensor(TO_CAMERA_AXES)[:, None, None]

    normals = torch.zeros((count, map_height, map_width, 3), device=device)
    for i, values, window, inside in crop_outputs(
        Sapiens2ForNormalEstimation, s.normal_model, normals_of
    ):
        normals[i][window] = torch.where(
            inside[..., None], F.normalize(values.permute(1, 2, 0), dim=-1), 0
        )

    def points_of(processor, model, images):
        inputs = processor(images=images, return_tensors="pt", device=device)
        out = model(**inputs.to(device, torch.bfloat16))
        if out.scales is None:
            return out.pointmaps
        return out.pointmaps / out.scales[:, :, None, None]

    point_depth = torch.full((count, map_height, map_width), torch.nan, device=device)
    point_focal = np.full(count, np.nan)
    for i, values, window, inside in crop_outputs(
        Sapiens2ForPointmapEstimation, s.pointmap_model, points_of
    ):
        point_depth[i][window] = torch.where(inside, values[2], torch.nan)
        # on the person only: the model's points around them are guesses
        point_focal[i] = pointmap_focal(values, inside) / factor

    processor = AutoImageProcessor.from_pretrained(s.depth_model)
    model = DepthProForDepthEstimation.from_pretrained(s.depth_model, dtype=torch.float16)
    model = model.to(device).eval()
    depth = torch.zeros((count, map_height, map_width), device=device)
    depth_focal = np.full(count, np.nan)
    for start in range(0, count, s.depth_batch):
        chunk = frames[start : start + s.depth_batch]
        inputs = processor(images=list(chunk), return_tensors="pt").to(device, torch.float16)
        results = processor.post_process_depth_estimation(
            model(**inputs), target_sizes=[(height, width)] * len(chunk)
        )
        for k, result in enumerate(results):
            depth[start + k] = F.interpolate(
                result["predicted_depth"].float()[None, None], (map_height, map_width), mode="area"
            )[0, 0]
            depth_focal[start + k] = float(result["focal_length"])
    del model
    torch.cuda.empty_cache()

    def half(values):
        return values.half().cpu().numpy()

    return SurfaceMaps(
        keys=np.asarray(keys, str),
        image_size=np.array([width, height]),
        map_size=np.array([map_width, map_height]),
        normals=half(normals),
        point_depth=half(point_depth),
        point_focal=point_focal,
        depth=half(depth),
        depth_focal=depth_focal,
    )


def _place(values: torch.Tensor, crop, factor: float, width: int, height: int):
    """``(C, h, w)`` model outputs over a crop (x0, y0, x1, y1 in frame pixels) resized to the
    crop at the maps' size: the part inside the map, ``(C, wh, ww)``, and that window of the
    map. None where the crop misses the map."""
    x0, y0, x1, y1 = (round(c * factor) for c in crop)
    if x1 <= x0 or y1 <= y0:
        return None
    resized = F.interpolate(values[None], (y1 - y0, x1 - x0), mode="bilinear")[0]
    fx0, fy0, fx1, fy1 = max(x0, 0), max(y0, 0), min(x1, width), min(y1, height)
    if fx1 <= fx0 or fy1 <= fy0:
        return None
    window = (slice(fy0, fy1), slice(fx0, fx1))
    return resized[:, fy0 - y0 : fy1 - y0, fx0 - x0 : fx1 - x0], window


def pointmap_focal(points: torch.Tensor, selected: torch.Tensor | None = None) -> float:
    """The focal length, in pixels of the image they belong to, that projects ``(3, h, w)``
    camera points (x right, y down, z forward) onto their own pixels with one principal point:
    least squares on ``u = f x / z + cx``, ``v = f y / z + cy``, twice, the second time without
    the points the first fit misses by more than three times their median. ``selected``: an
    optional ``(h, w)`` bool choice of points. NaN with fewer than 100 points in front of the
    camera."""
    x, y, z = points
    height, width = z.shape
    v, u = torch.meshgrid(
        torch.arange(height, device=z.device, dtype=z.dtype),
        torch.arange(width, device=z.device, dtype=z.dtype),
        indexing="ij",
    )
    keep = torch.isfinite(points).all(dim=0) & (z > 0)
    if selected is not None:
        keep &= selected
    if keep.sum() >= 100:
        keep &= z > 0.3 * z[keep].median()
    if keep.sum() < 100:
        return float("nan")
    a, b, pixels = x[keep] / z[keep], y[keep] / z[keep], torch.cat((u[keep], v[keep]))
    ones, zeros = torch.ones_like(a), torch.zeros_like(a)
    design = torch.stack(
        (torch.cat((a, b)), torch.cat((ones, zeros)), torch.cat((zeros, ones))), dim=1
    ).double()
    target = pixels.double()
    used = torch.ones_like(target, dtype=torch.bool)
    for _ in range(2):
        solution = torch.linalg.lstsq(design[used], target[used, None]).solution[:, 0]
        miss = (design @ solution - target).abs()
        used = miss <= 3 * miss.median().clamp_min(1e-6)
    focal = float(solution[0])
    return focal if focal > 0 else float("nan")
