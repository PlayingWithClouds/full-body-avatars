"""Training the avatar's Gaussians on the capture."""

from __future__ import annotations

from pathlib import Path

from avatar_splats.body_models.anny import AnnyBody
from avatar_splats.body_models.hybrid import Avatar
from avatar_splats.body_models.ict import IctFaceKit
from avatar_splats.config import Config
from avatar_splats.fitting import HEAD_VIDEOS, BodyFit, HeadFit, fitted_path
from avatar_splats.preprocess import Preprocessed, preprocessed_path
from avatar_splats.training.posing import Posing
from avatar_splats.training.textures import TextureRecord, Textures, bake_textures
from avatar_splats.training.train import train, training_views
from avatar_splats.training.views import View, body_views, head_views


def load_views(config: Config) -> list[View]:
    """Every frame of the capture, at the training resolution."""
    device = config.training.device
    resolution = config.training.resolution
    views = body_views(
        config.capture.body,
        Preprocessed.load(preprocessed_path(config, "body")),
        BodyFit.load(fitted_path(config, "body")),
        resolution,
        device,
    )
    head = IctFaceKit(config.body_models.ict_facekit).to(device)
    for video in HEAD_VIDEOS:
        views += head_views(
            video,
            getattr(config.capture, video),
            Preprocessed.load(preprocessed_path(config, video)),
            HeadFit.load(fitted_path(config, video)),
            head,
            resolution,
        )
    return views


def load_posing(config: Config, avatar: Avatar) -> Posing:
    """The avatar posed as the fits found it, with no corrections yet."""
    return Posing(
        avatar,
        AnnyBody(),
        IctFaceKit(config.body_models.ict_facekit),
        BodyFit.load(fitted_path(config, "body")),
        {video: HeadFit.load(fitted_path(config, video)) for video in HEAD_VIDEOS},
    ).to(config.training.device)


def bake(config: Config, avatar: Avatar, views: list[View], posing: Posing) -> TextureRecord:
    """The avatar's textures baked from the training frames, before training."""
    return bake_textures(avatar, training_views(views, config.training), posing, config.training)


def train_capture(config: Config) -> dict[str, Path]:
    """Train the textures and Gaussians and return where they and their evaluation were written."""
    avatar = Avatar.load(fitted_path(config, "avatar"))
    views = load_views(config)
    posing = load_posing(config, avatar)
    textures = Textures(bake(config, avatar, views, posing)).to(config.training.device)
    scene, colours, evaluation = train(views, posing, avatar, textures, config.training)
    written = {
        "gaussians": trained_path(config, "gaussians"),
        "textures": trained_path(config, "textures"),
        "colours": trained_path(config, "colours"),
        "evaluation": trained_path(config, "evaluation"),
    }
    scene.gaussians.record().save(written["gaussians"])
    scene.textures.record().save(written["textures"])
    colours.record().save(written["colours"])
    evaluation.save(written["evaluation"])
    return written


def trained_path(config: Config, name: str) -> Path:
    return config.outputs / "training" / f"{name}.npz"
