"""Photometric optimization of the bound Gaussians."""

from avatar_splats.training.pipeline import load_views, train_capture, trained_path
from avatar_splats.training.train import Evaluation, render_view, target, train

__all__ = [
    "Evaluation",
    "load_views",
    "render_view",
    "target",
    "train",
    "train_capture",
    "trained_path",
]
