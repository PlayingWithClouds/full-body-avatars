"""Finding the performer in a frame, to prompt the segmentation."""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection


@torch.inference_mode()
def detect_person(image: np.ndarray, model_id: str, device: str) -> np.ndarray:
    """Return the box of the most confident person in an RGB image, as x0, y0, x1, y1.

    Raises:
        LookupError: when no person is detected.
    """
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForObjectDetection.from_pretrained(model_id).to(device).eval()

    inputs = processor(images=image, return_tensors="pt").to(device)
    detections = processor.post_process_object_detection(
        model(**inputs), target_sizes=[image.shape[:2]], threshold=0.5
    )[0]

    people = detections["labels"] == model.config.label2id["person"]
    if not people.any():
        raise LookupError("no person detected")
    best = detections["scores"][people].argmax()
    return detections["boxes"][people][best].cpu().numpy()
