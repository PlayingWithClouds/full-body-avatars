"""Fitting body shape, per-frame pose and facial expressions to the preprocessed frames."""

from avatar_splats.fitting.body import BodyFit, fit_body
from avatar_splats.fitting.head import HeadFit, fit_heads, head_vertices
from avatar_splats.fitting.pipeline import HEAD_VIDEOS, fit_capture, fitted_path

__all__ = [
    "HEAD_VIDEOS",
    "BodyFit",
    "HeadFit",
    "fit_body",
    "fit_capture",
    "fit_heads",
    "fitted_path",
    "head_vertices",
]
