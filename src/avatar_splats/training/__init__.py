"""Photometric optimization of the avatar's textures and bound Gaussians."""

from avatar_splats.training.colour import ColourCorrection, ColourRecord
from avatar_splats.training.pipeline import (
    bake,
    load_posing,
    load_views,
    train_capture,
    trained_path,
)
from avatar_splats.training.posing import Posing
from avatar_splats.training.textures import TextureRecord, Textures, bake_textures, fill_holes
from avatar_splats.training.train import (
    Evaluation,
    Scene,
    evaluate,
    place_gaussians,
    scored_pixels,
    target,
    train,
)

__all__ = [
    "ColourCorrection",
    "ColourRecord",
    "Evaluation",
    "Posing",
    "Scene",
    "TextureRecord",
    "Textures",
    "bake",
    "bake_textures",
    "fill_holes",
    "load_posing",
    "load_views",
    "evaluate",
    "place_gaussians",
    "scored_pixels",
    "target",
    "train",
    "train_capture",
    "trained_path",
]
