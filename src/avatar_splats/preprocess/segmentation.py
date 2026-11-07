"""Segmenting the performer through a video with SAM2."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from transformers import Sam2VideoModel, Sam2VideoProcessor


@torch.inference_mode()
def track_person(
    frames: Sequence[np.ndarray], box: np.ndarray, model_id: str, device: str
) -> np.ndarray:
    """Segment the person inside ``box`` on the first frame, and follow them through the rest.

    All frames are held in host memory while tracking, so long videos should be subsampled.

    Returns:
        ``(N, H, W)`` bool masks, one per frame.
    """
    processor = Sam2VideoProcessor.from_pretrained(model_id)
    model = Sam2VideoModel.from_pretrained(model_id).to(device).eval()

    session = processor.init_video_session(
        video=list(frames), inference_device=device, video_storage_device="cpu"
    )
    processor.add_inputs_to_inference_session(
        inference_session=session, frame_idx=0, obj_ids=1, input_boxes=[[box.tolist()]]
    )

    height, width = frames[0].shape[:2]
    masks = np.zeros((len(frames), height, width), dtype=bool)
    for output in model.propagate_in_video_iterator(session, start_frame_idx=0):
        mask = processor.post_process_masks([output.pred_masks], original_sizes=[(height, width)])
        masks[output.frame_idx] = mask[0][0, 0].cpu().numpy()
    return masks


def mask_boxes(masks: np.ndarray) -> np.ndarray:
    """Bounding boxes of ``(N, H, W)`` masks as ``(N, 4)`` x0, y0, x1, y1; NaN for empty masks."""
    boxes = np.full((len(masks), 4), np.nan)
    for i, mask in enumerate(masks):
        rows = np.flatnonzero(mask.any(axis=1))
        cols = np.flatnonzero(mask.any(axis=0))
        if rows.size:
            boxes[i] = cols[0], rows[0], cols[-1] + 1, rows[-1] + 1
    return boxes
