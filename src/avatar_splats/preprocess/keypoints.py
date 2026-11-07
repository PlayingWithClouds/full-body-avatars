"""Body keypoints with ViTPose."""

from __future__ import annotations

from collections.abc import Sequence

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


@torch.inference_mode()
def estimate_keypoints(
    frames: Sequence[np.ndarray],
    boxes: np.ndarray,
    model_id: str,
    device: str,
    batch_size: int = 16,
) -> np.ndarray:
    """Estimate the COCO keypoints of the person inside each frame's box (x0, y0, x1, y1).

    Returns:
        ``(N, 17, 3)`` keypoints as x, y in pixels and confidence; NaN where the box is NaN.
    """
    processor = AutoProcessor.from_pretrained(model_id)
    model = VitPoseForPoseEstimation.from_pretrained(model_id).to(device).eval()

    keypoints = np.full((len(frames), len(COCO_KEYPOINTS), 3), np.nan)
    valid = np.flatnonzero(~np.isnan(boxes).any(axis=1))
    for start in range(0, len(valid), batch_size):
        batch = valid[start : start + batch_size]
        images = [frames[i] for i in batch]
        coco_boxes = [[[x0, y0, x1 - x0, y1 - y0]] for x0, y0, x1, y1 in boxes[batch]]

        inputs = processor(images, boxes=coco_boxes, return_tensors="pt").to(device)
        results = processor.post_process_pose_estimation(model(**inputs), boxes=coco_boxes)
        for i, (person,) in zip(batch, results, strict=True):
            keypoints[i, :, :2] = person["keypoints"].cpu().numpy()
            keypoints[i, :, 2] = person["scores"].cpu().numpy()
    return keypoints
