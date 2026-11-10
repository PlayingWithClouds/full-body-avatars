"""Extracting motion from a video: observe the people, fit the avatar's body to each, write the
clip."""

from __future__ import annotations

from pathlib import Path

from avatar_splats.body_models.anny import AnnyBody
from avatar_splats.body_models.hybrid import Avatar
from avatar_splats.config import Config
from avatar_splats.fitting.body import BodyFit
from avatar_splats.fitting.pipeline import fitted_path
from avatar_splats.motion.clip import motion_clip, write_motion
from avatar_splats.motion.fit import fit_motion
from avatar_splats.motion.observe import Observations, observe_people

RESULTS = ("observations", "fit", "clip")
"""What the stage writes per video, in order."""


def extract_motion(config: Config, video: Path, name: str, refit: bool = False) -> dict[str, Path]:
    """Extract the motion of everyone in ``video`` as a clip for the avatar.

    Args:
        name: names the results, under ``outputs/motion/<name>/``.
        refit: start from the observations of an earlier run instead of the video.

    Returns:
        Where each result was written.
    """
    written = {result: motion_path(config, name, result) for result in RESULTS}
    if refit:
        observations = Observations.load(written["observations"])
    else:
        observations = observe_people(video, config.motion, config.preprocess)
        observations.save(written["observations"])

    body = AnnyBody().to(config.motion.device)
    avatar = Avatar.load(fitted_path(config, "avatar"))
    fit = fit_motion(
        observations,
        body,
        BodyFit.load(fitted_path(config, "body")),
        list(avatar.expression_names),
        config.motion,
        config.fitting.silhouette_inset,
    )
    fit.save(written["fit"])
    clip = motion_clip(fit, body, avatar)
    clip.save(written["clip"])
    written["motion"] = motion_file_path(config, name)
    write_motion(written["motion"], clip)
    return written


def motion_file_path(config: Config, name: str) -> Path:
    """Where the ``.motion`` file of a motion video is written, for the viewer."""
    return config.outputs / "motion" / name / f"{name}.motion"


def motion_path(config: Config, name: str, result: str) -> Path:
    """Where a result (one of ``RESULTS``) of a motion video is stored."""
    return config.outputs / "motion" / name / f"{result}.npz"
