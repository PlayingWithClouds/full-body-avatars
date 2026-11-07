"""Training the avatar's Gaussians on the capture."""

from __future__ import annotations

from pathlib import Path

from avatar_splats.body_models.anny import AnnyBody
from avatar_splats.body_models.hybrid import Avatar, Rig
from avatar_splats.body_models.ict import IctFaceKit
from avatar_splats.config import Config
from avatar_splats.fitting import HEAD_VIDEOS, BodyFit, HeadFit, fitted_path
from avatar_splats.preprocess import Preprocessed, preprocessed_path
from avatar_splats.training.train import train
from avatar_splats.training.views import View, body_views, head_views


def load_views(config: Config, avatar: Avatar) -> list[View]:
    """Every frame of the capture, with the avatar posed as the fits found it."""
    device = config.training.device
    resolution = config.training.resolution
    views = body_views(
        config.capture.body,
        Preprocessed.load(preprocessed_path(config, "body")),
        BodyFit.load(fitted_path(config, "body")),
        Rig(avatar).to(device),
        AnnyBody().to(device),
        resolution,
    )
    head = IctFaceKit(config.body_models.ict_facekit).to(device)
    for video in HEAD_VIDEOS:
        views += head_views(
            video,
            getattr(config.capture, video),
            Preprocessed.load(preprocessed_path(config, video)),
            HeadFit.load(fitted_path(config, video)),
            avatar,
            head,
            resolution,
        )
    return views


def train_capture(config: Config) -> dict[str, Path]:
    """Train the Gaussians and return where they and their evaluation were written."""
    avatar = Avatar.load(fitted_path(config, "avatar"))
    gaussians, evaluation = train(load_views(config, avatar), avatar, config.training)
    written = {
        "gaussians": trained_path(config, "gaussians"),
        "evaluation": trained_path(config, "evaluation"),
    }
    gaussians.record().save(written["gaussians"])
    evaluation.save(written["evaluation"])
    return written


def trained_path(config: Config, name: str) -> Path:
    return config.outputs / "training" / f"{name}.npz"
