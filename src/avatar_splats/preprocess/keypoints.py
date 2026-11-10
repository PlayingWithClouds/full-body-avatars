"""Body keypoints with ViTPose, and with Sapiens2 where they matter most."""

from __future__ import annotations

from collections.abc import Sequence
from functools import cache

import numpy as np
import torch
from transformers import AutoProcessor, VitPoseForPoseEstimation

COCO_KEYPOINTS = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


@cache
def load_vitpose(
    model_id: str, device: str, compiled: bool = False
) -> tuple[AutoProcessor, VitPoseForPoseEstimation]:
    """ViTPose's processor and model, for ``estimate_keypoints``' ``models``: loaded (and
    compiled) once for many calls."""
    processor = AutoProcessor.from_pretrained(model_id)
    model = VitPoseForPoseEstimation.from_pretrained(model_id).to(device).eval()
    return processor, torch.compile(model) if compiled else model


@torch.inference_mode()
def estimate_keypoints(
    frames: Sequence[np.ndarray],
    boxes: np.ndarray,
    model_id: str,
    device: str,
    batch_size: int = 16,
    compiled: bool = False,
    models: tuple | None = None,
) -> np.ndarray:
    """Estimate the COCO keypoints of the person inside each frame's box (x0, y0, x1, y1).

    The person is cropped from the frame on the GPU (``person_crops``), as ViTPose's processor
    crops it on the CPU. ``compiled`` runs the model through ``torch.compile``; every batch is
    then padded to ``batch_size`` so it compiles once. ``models``: ``load_vitpose``'s, loaded
    here where not given.

    Returns:
        ``(N, 17, 3)`` keypoints as x, y in pixels and confidence; NaN where the box is NaN.
    """
    processor, model = models or load_vitpose(model_id, device, compiled)

    keypoints = np.full((len(frames), len(COCO_KEYPOINTS), 3), np.nan)
    valid = np.flatnonzero(~np.isnan(boxes).any(axis=1))
    for start in range(0, len(valid), batch_size):
        batch = valid[start : start + batch_size]
        coco_boxes = [[[x0, y0, x1 - x0, y1 - y0]] for x0, y0, x1, y1 in boxes[batch]]
        images = [torch.from_numpy(np.ascontiguousarray(frames[i])).to(device) for i in batch]
        pixel_values = person_crops(images, [box[0] for box in coco_boxes], processor)
        count = len(pixel_values)
        if compiled and count < batch_size:
            pad = pixel_values[-1:].expand(batch_size - count, -1, -1, -1)
            pixel_values = torch.cat([pixel_values, pad])
        outputs = model(pixel_values=pixel_values)
        outputs.heatmaps = outputs.heatmaps[:count]
        results = processor.post_process_pose_estimation(outputs, boxes=coco_boxes)
        for i, (person,) in zip(batch, results, strict=True):
            keypoints[i, :, :2] = person["keypoints"].cpu().numpy()
            keypoints[i, :, 2] = person["scores"].cpu().numpy()
    return keypoints


def person_crops(
    images: Sequence[torch.Tensor], boxes: Sequence[Sequence[float]], processor
) -> torch.Tensor:
    """``(B, 3, h, w)`` ViTPose's normalized input: each ``(H, W, 3)`` uint8 image's box (x, y,
    width, height) cropped as its processor crops it, by the same warp, but sampled on the
    image's device (bilinear, zero outside the image, as ``cv2.warpAffine``)."""
    from transformers.models.vitpose.image_processing_vitpose import (
        box_to_center_and_scale,
        get_warp_matrix,
    )

    size = processor.size
    crops = []
    for image, box in zip(images, boxes, strict=True):
        center, scale = box_to_center_and_scale(
            box, image_width=size.width, image_height=size.height, normalize_factor=200.0
        )
        warp = get_warp_matrix(
            0, center * 2.0, np.array((size.width, size.height)) - 1.0, scale * 200.0
        )  # (2, 3): image pixels to crop pixels
        inverse = np.linalg.inv(np.vstack([warp, [0, 0, 1]]))[:2]
        inverse = torch.from_numpy(inverse).float().to(image.device)
        ys, xs = torch.meshgrid(
            torch.arange(size.height, device=image.device, dtype=torch.float32),
            torch.arange(size.width, device=image.device, dtype=torch.float32),
            indexing="ij",
        )
        source = torch.stack([xs, ys, torch.ones_like(xs)], -1) @ inverse.T  # (h, w, 2)
        height, width = image.shape[:2]
        grid = torch.stack(
            [(2 * source[..., 0] + 1) / width - 1, (2 * source[..., 1] + 1) / height - 1], -1
        )
        picture = image.permute(2, 0, 1)[None].float()
        crops.append(
            torch.nn.functional.grid_sample(
                picture, grid[None], mode="bilinear", padding_mode="zeros", align_corners=False
            )[0]
        )
    crops = torch.stack(crops) / 255
    mean = torch.tensor(processor.image_mean, device=crops.device)[:, None, None]
    std = torch.tensor(processor.image_std, device=crops.device)[:, None, None]
    return (crops - mean) / std


COCO_FROM_GOLIATH = (0, 1, 2, 3, 4, 5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14)
"""The COCO keypoints among Sapiens2's 308 (Goliath's ``GOLIATH_KEYPOINTS``: the body's 15
without the wrists, the feet's 6, the right hand's 21 ending at its wrist, the left's, 7 on the
arms and neck, then the face), checked against its mirror pairs and on frames."""
EXTRA_KEYPOINTS = (
    "left_big_toe", "left_small_toe", "left_heel", "right_big_toe", "right_small_toe",
    "right_heel", "left_olecranon", "right_olecranon", "left_cubital_fossa",
    "right_cubital_fossa", "left_acromion", "right_acromion",
)  # fmt: skip
"""Goliath's keypoints beyond COCO's that the fit uses: on the feet (which way they point), at
the back and front of the elbows (which way the arm is twisted) and on top of the shoulders."""
EXTRA_FROM_GOLIATH = (15, 16, 17, 18, 19, 20, 63, 64, 65, 66, 67, 68)
_HAND_FROM_GOLIATH = (20, 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 19, 18, 17, 16)
"""MediaPipe's 21 hand landmarks (the wrist, then each finger from the knuckle nearest it to the
tip) within one of Goliath's hands (each finger from the tip back, the wrist last)."""
GOLIATH_HANDS = (42, 21)
"""Where the left and the right hand start among Goliath's keypoints."""


def coco_keypoints(goliath: np.ndarray) -> np.ndarray:
    """``(..., 17, 3)`` COCO keypoints of ``(..., 308, 3)`` Goliath keypoints."""
    return goliath[..., COCO_FROM_GOLIATH, :]


def extra_keypoints(goliath: np.ndarray) -> np.ndarray:
    """``(..., 12, 3)`` the ``EXTRA_KEYPOINTS`` of ``(..., 308, 3)`` Goliath keypoints."""
    return goliath[..., EXTRA_FROM_GOLIATH, :]


def hand_landmarks(goliath: np.ndarray) -> np.ndarray:
    """``(..., 2, 21, 3)`` the left and right hand in MediaPipe's order, of ``(..., 308, 3)``
    Goliath keypoints."""
    return np.stack(
        [goliath[..., [start + i for i in _HAND_FROM_GOLIATH], :] for start in GOLIATH_HANDS],
        axis=-3,
    )


@torch.inference_mode()
def estimate_sapiens_keypoints(
    frames: Sequence[np.ndarray],
    boxes: np.ndarray,
    model_id: str,
    device: str,
    batch_size: int = 8,
) -> np.ndarray:
    """All of Sapiens2's keypoints of the person inside each frame's box (x0, y0, x1, y1), by its
    pose model: slower than ViTPose, but it finds limbs ViTPose gives up on (a knee drawn up
    against the chest it put on the other knee). ``coco_keypoints``, ``extra_keypoints`` and
    ``hand_landmarks`` pick out what the fit uses.

    Returns:
        ``(N, 308, 3)`` Goliath keypoints as x, y in pixels and confidence; NaN where the box
        is NaN.
    """
    from transformers import AutoImageProcessor, Sapiens2ForPoseEstimation

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = Sapiens2ForPoseEstimation.from_pretrained(model_id, dtype=torch.bfloat16)
    model = model.to(device).eval()
    keypoints = np.full((len(frames), model.config.num_labels, 3), np.nan)
    valid = np.flatnonzero(~np.isnan(boxes).any(axis=1))
    for start in range(0, len(valid), batch_size):
        batch = valid[start : start + batch_size]
        coco_boxes = [[[x0, y0, x1 - x0, y1 - y0]] for x0, y0, x1, y1 in boxes[batch]]
        inputs = processor(images=[frames[i] for i in batch], boxes=coco_boxes, return_tensors="pt")
        outputs = model(**inputs.to(device, torch.bfloat16))
        results = processor.post_process_pose_estimation(outputs, boxes=coco_boxes)
        for i, (person,) in zip(batch, results, strict=True):
            keypoints[i, :, :2] = person["keypoints"].float().cpu().numpy()
            keypoints[i, :, 2] = person["scores"].float().cpu().numpy()
    del model
    torch.cuda.empty_cache()
    return keypoints
