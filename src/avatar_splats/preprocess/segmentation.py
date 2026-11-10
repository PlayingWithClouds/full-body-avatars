"""Segmenting people through a video with SAM2."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache

import numpy as np
import torch
from transformers import Sam2VideoModel, Sam2VideoProcessor


@cache
def _load(model_id: str, device: str) -> tuple[Sam2VideoProcessor, Sam2VideoModel]:
    processor = Sam2VideoProcessor.from_pretrained(model_id)
    return processor, Sam2VideoModel.from_pretrained(model_id).to(device).eval()


def track_person(
    frames: Sequence[np.ndarray], box: np.ndarray, model_id: str, device: str
) -> np.ndarray:
    """Segment the person inside ``box`` on the first frame, and follow them through the rest.

    Returns:
        ``(N, H, W)`` bool masks, one per frame.
    """
    return track_people(frames, 0, model_id, device, boxes=box[None])[0]


def track_people(
    frames: Sequence[np.ndarray],
    prompt_frame: int,
    model_id: str,
    device: str,
    *,
    boxes: np.ndarray | None = None,
    masks: np.ndarray | None = None,
) -> np.ndarray:
    """Segment people on frame ``prompt_frame``, and follow each through the frames before and
    after it.

    Each person is given by a box, or by a mask (``(P, H, W)`` bool), as when going on from
    where an earlier stretch of the video left off; an empty mask falls back to the box at the
    same index. SAM2 tracks the people together, each with a memory of its own, so a person keeps
    their mask while someone else passes in front. All frames are held in host memory while
    tracking, so long videos should be tracked in stretches.

    Args:
        boxes: ``(P, 4)`` x0, y0, x1, y1 (more columns are ignored).

    Returns:
        ``(P, N, H, W)`` bool masks, one per person and frame.
    """
    return track_people_scored(
        frames, prompt_frame, model_id, device, boxes=boxes, masks=masks
    ).masks


@dataclass(frozen=True)
class Tracked:
    """People followed through frames by ``track_people_scored``."""

    masks: np.ndarray
    """``(P, N, H, W)`` bool masks, one per person and frame."""
    presence: np.ndarray
    """``(P, N)`` SAM2's probability that the person shows in the frame at all."""
    certainty: np.ndarray
    """``(P, N)`` the mean probability SAM2 gives the pixels of each mask; NaN where empty."""


@torch.inference_mode()
def track_people_scored(
    frames: Sequence[np.ndarray],
    prompt_frame: int,
    model_id: str,
    device: str,
    *,
    boxes: np.ndarray | None = None,
    masks: np.ndarray | None = None,
) -> Tracked:
    """``track_people``, with SAM2's own judgement of each mask: whether the person is there at
    all, and how sure it is of the pixels it gave them."""
    processor, model = _load(model_id, device)
    count = len(boxes) if boxes is not None else len(masks)
    session = processor.init_video_session(
        video=list(frames), inference_device=device, video_storage_device="cpu"
    )
    for person in range(count):
        prompt = {}
        if masks is not None and masks[person].any():
            prompt["input_masks"] = [masks[person]]
        else:
            prompt["input_boxes"] = [[boxes[person][:4].tolist()]]
        processor.add_inputs_to_inference_session(
            inference_session=session, frame_idx=prompt_frame, obj_ids=person + 1, **prompt
        )
        # Each prompt is segmented before the next is added, as SAM2 conditions on it.
        model(inference_session=session, frame_idx=prompt_frame)

    height, width = frames[0].shape[:2]
    found = np.zeros((count, len(frames), height, width), dtype=bool)
    presence = np.zeros((count, len(frames)))
    certainty = np.full((count, len(frames)), np.nan)
    for reverse in (False, True):
        if reverse and prompt_frame == 0:
            break
        outputs = model.propagate_in_video_iterator(
            session, start_frame_idx=prompt_frame, reverse=reverse
        )
        for output in outputs:
            logits = processor.post_process_masks(
                [output.pred_masks], original_sizes=[(height, width)], binarize=False
            )[0][:, 0]
            mask = logits > 0
            found[:, output.frame_idx] = mask.cpu().numpy()
            presence[:, output.frame_idx] = output.object_score_logits.sigmoid().flatten().cpu()
            probability = logits.sigmoid()
            for person in range(count):
                if mask[person].any():
                    sure = probability[person][mask[person]].mean()
                    certainty[person, output.frame_idx] = sure.item()
    return Tracked(found, presence, certainty)


def mask_boxes(masks: np.ndarray) -> np.ndarray:
    """Bounding boxes of ``(N, H, W)`` masks as ``(N, 4)`` x0, y0, x1, y1; NaN for empty masks."""
    boxes = np.full((len(masks), 4), np.nan)
    for i, mask in enumerate(masks):
        rows = np.flatnonzero(mask.any(axis=1))
        cols = np.flatnonzero(mask.any(axis=0))
        if rows.size:
            boxes[i] = cols[0], rows[0], cols[-1] + 1, rows[-1] + 1
    return boxes
