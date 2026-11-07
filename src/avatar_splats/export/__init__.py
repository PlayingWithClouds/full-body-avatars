"""The ``.avatar`` file read by the game runtime, and a reference runtime in Python."""

from pathlib import Path

from avatar_splats.body_models.hybrid import Avatar
from avatar_splats.config import Config
from avatar_splats.export.avatar_file import AvatarFile, avatar_arrays, read_avatar, write_avatar
from avatar_splats.export.runtime import AvatarRuntime, bind_relative, forward_kinematics
from avatar_splats.fitting.pipeline import fitted_path
from avatar_splats.gaussians import GaussianRecord
from avatar_splats.training.pipeline import trained_path

__all__ = [
    "AvatarFile",
    "AvatarRuntime",
    "avatar_arrays",
    "bind_relative",
    "export_capture",
    "exported_path",
    "forward_kinematics",
    "read_avatar",
    "write_avatar",
]


def export_capture(config: Config) -> Path:
    """Write the trained avatar as an ``.avatar`` file and return its path."""
    avatar = Avatar.load(fitted_path(config, "avatar"))
    gaussians = GaussianRecord.load(trained_path(config, "gaussians"))
    path = exported_path(config)
    write_avatar(path, avatar_arrays(avatar, gaussians))
    return path


def exported_path(config: Config) -> Path:
    return config.outputs / "export" / "performer.avatar"
