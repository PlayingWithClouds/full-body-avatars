"""MediaPipe's ARKit blendshapes as the performer's expressions.

MediaPipe's weights are for an average face: a neutral performer scores some of them above zero
(a resting mouth is never quite ``mouthClose``). Her neutral frames, the frontal ones of the
neutral head video, give each blendshape's baseline; subtracting it and stretching the rest to
one puts her neutral face at zero.
"""

from __future__ import annotations

import numpy as np


def neutral_baseline(blendshapes: np.ndarray, yaw: np.ndarray, max_yaw: float) -> np.ndarray:
    """``(B,)`` the median of each blendshape over the frames that look at the camera.

    Args:
        blendshapes: ``(N, B)`` weights of a neutral head video; NaN where no face was found.
        yaw: ``(N,)`` the head's yaw in radians, zero when frontal.
        max_yaw: how far from frontal, in radians, a frame still counts.

    Raises:
        ValueError: when no frontal frame has a face.
    """
    frontal = (np.abs(yaw) <= max_yaw) & ~np.isnan(blendshapes).any(axis=1)
    if not frontal.any():
        raise ValueError("no frontal frame with a face to take the neutral face from")
    return np.median(blendshapes[frontal], axis=0)


def calibrate(blendshapes: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """``(N, B)`` the weights with the baseline taken off and the rest stretched to one, in
    [0, 1]; NaN stays NaN."""
    stretched = (blendshapes - baseline) / np.maximum(1.0 - baseline, 1e-3)
    return np.clip(stretched, 0.0, 1.0)
