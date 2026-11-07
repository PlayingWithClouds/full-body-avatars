"""Fitting the capture: the body, the heads, and the avatar joining them."""

from __future__ import annotations

from pathlib import Path

from avatar_splats.body_models.anny import AnnyBody
from avatar_splats.body_models.hybrid import build_avatar
from avatar_splats.body_models.ict import IctFaceKit
from avatar_splats.config import VIDEOS, Config
from avatar_splats.data import probe
from avatar_splats.fitting.body import fit_body
from avatar_splats.fitting.head import fit_heads
from avatar_splats.preprocess import Preprocessed, preprocessed_path

HEAD_VIDEOS = tuple(video for video in VIDEOS if video != "body")


def fit_capture(config: Config) -> dict[str, Path]:
    """Fit the body and head videos, build the avatar, and return where each result was written."""
    device = config.fitting.device
    preprocessed = {video: Preprocessed.load(preprocessed_path(config, video)) for video in VIDEOS}
    sizes = {}
    for video in VIDEOS:
        info = probe(getattr(config.capture, video))
        sizes[video] = (info.width, info.height)

    body = AnnyBody().to(device)
    head = IctFaceKit(config.body_models.ict_facekit)
    written = {}

    body_fit = fit_body(preprocessed["body"], sizes["body"], body, config.fitting)
    written["body"] = fitted_path(config, "body")
    body_fit.save(written["body"])

    head_fits = fit_heads(
        {v: (preprocessed[v], sizes[v]) for v in HEAD_VIDEOS}, head, config.fitting
    )
    for video, fit in head_fits.items():
        written[video] = fitted_path(config, video)
        fit.save(written[video])

    identity = next(iter(head_fits.values())).identity
    avatar = build_avatar(
        body, body_fit.phenotype, body_fit.offsets, head, identity, config.body_models
    )
    written["avatar"] = fitted_path(config, "avatar")
    avatar.save(written["avatar"])
    return written


def fitted_path(config: Config, name: str) -> Path:
    """Where the fit of a video, or the avatar (``name="avatar"``), is stored."""
    return config.outputs / "fitting" / f"{name}.npz"
