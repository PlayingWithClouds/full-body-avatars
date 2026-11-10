"""Finding people in a frame, to prompt the segmentation."""

from __future__ import annotations

from collections.abc import Sequence
from functools import cache

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection


def detect_person(image: np.ndarray, model_id: str, device: str) -> np.ndarray:
    """Return the box of the most confident person in an RGB image, as x0, y0, x1, y1.

    Raises:
        LookupError: when no person is detected.
    """
    (boxes,) = detect_people([image], model_id, device, threshold=0.5)
    if not len(boxes):
        raise LookupError("no person detected")
    return boxes[0]


@cache
def load_detector(
    model_id: str, device: str, compiled: bool = False
) -> tuple[AutoImageProcessor, AutoModelForObjectDetection]:
    """The detector's processor and model, for ``detect_people``'s ``models``: loaded (and
    compiled) once for many calls, as loading and compiling take seconds."""
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForObjectDetection.from_pretrained(model_id).to(device).eval()
    if not compiled:
        return processor, model
    # RT-DETR's attention checks a tensor's value; without this every check breaks the graph.
    torch._dynamo.config.capture_scalar_outputs = True
    return processor, torch.compile(model)


@torch.inference_mode()
def detect_people(
    images: Sequence[np.ndarray],
    model_id: str,
    device: str,
    threshold: float,
    batch_size: int = 16,
    compiled: bool = False,
    models: tuple | None = None,
) -> list[np.ndarray]:
    """Find the people in RGB images.

    The images go to the GPU as they are and are resized there, ``batch_size`` at a time: on
    the CPU, one at a time, preprocessing kept the GPU waiting. ``compiled`` runs the model
    through ``torch.compile`` (worth it for many images of one size); every batch is then
    padded to ``batch_size`` so it compiles once. ``models``: ``load_detector``'s, loaded
    here where not given.

    Returns:
        Per image, ``(K, 5)`` boxes as x0, y0, x1, y1 in pixels and confidence, the most
        confident first; only those at least ``threshold`` confident.
    """
    processor, model = models or load_detector(model_id, device, compiled)
    person = model.config.label2id["person"]

    found = []
    for start in range(0, len(images), batch_size):
        chunk = images[start : start + batch_size]
        tensors = [
            torch.from_numpy(np.ascontiguousarray(image)).to(device).permute(2, 0, 1)
            for image in chunk
        ]
        if compiled:
            tensors += [tensors[-1]] * (batch_size - len(tensors))
        inputs = processor(images=tensors, return_tensors="pt", device=device).to(device)
        detections = processor.post_process_object_detection(
            model(**inputs),
            target_sizes=[image.shape[:2] for image in chunk]
            + [chunk[-1].shape[:2]] * (len(tensors) - len(chunk)),
            threshold=threshold,
        )[: len(chunk)]
        for result in detections:
            people = result["labels"] == person
            boxes = (
                torch.cat((result["boxes"][people], result["scores"][people, None]), dim=1)
                .cpu()
                .numpy()
            )
            found.append(boxes[np.argsort(-boxes[:, 4])])
    return found
