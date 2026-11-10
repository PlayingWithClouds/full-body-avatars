"""The ``.avatar`` file read by the game runtime, and a reference runtime in Python."""

from pathlib import Path

import numpy as np

from avatar_splats.body_models.hybrid import Avatar
from avatar_splats.body_models.ict import landmark_vertices
from avatar_splats.config import Config
from avatar_splats.data import read_frames
from avatar_splats.export.avatar_file import AvatarFile, avatar_arrays, read_avatar, write_avatar
from avatar_splats.export.eyes import EyeLook, eye_metadata, measure_eyes
from avatar_splats.export.face_hair import (
    FaceHair,
    face_hair_metadata,
    grow_face_hair,
    measure_hair_colours,
)
from avatar_splats.export.mouth import expose_lower_teeth
from avatar_splats.export.runtime import AvatarRuntime, bind_relative, forward_kinematics
from avatar_splats.fitting.head import HeadFit
from avatar_splats.fitting.pipeline import HEAD_VIDEOS, fitted_path
from avatar_splats.gaussians import GaussianRecord
from avatar_splats.preprocess import Preprocessed, preprocessed_path
from avatar_splats.training.pipeline import trained_path
from avatar_splats.training.textures import TextureRecord

__all__ = [
    "AvatarFile",
    "AvatarRuntime",
    "EyeLook",
    "FaceHair",
    "avatar_arrays",
    "bind_relative",
    "eye_metadata",
    "face_hair_metadata",
    "grow_face_hair",
    "export_capture",
    "exported_path",
    "forward_kinematics",
    "measure_eyes",
    "measure_hair_colours",
    "read_avatar",
    "write_avatar",
]


def export_capture(config: Config) -> Path:
    """Write the trained avatar as an ``.avatar`` file and return its path."""
    avatar = expose_lower_teeth(Avatar.load(fitted_path(config, "avatar")))
    gaussians = GaussianRecord.load(trained_path(config, "gaussians"))
    textures = TextureRecord.load(trained_path(config, "textures"))
    preprocessed = Preprocessed.load(preprocessed_path(config, "head_static"))
    frames = np.stack(list(read_frames(config.capture.head_static)))[preprocessed.frame_indices]
    look = measure_eyes(frames, preprocessed.face_landmarks)
    eyes = eye_metadata(avatar, look, landmark_vertices(config.body_models.ict_facekit))
    identity = HeadFit.load(fitted_path(config, HEAD_VIDEOS[0])).identity  # the avatar's head's
    hair = grow_face_hair(
        avatar, textures, config.body_models.ict_facekit, identity, np.random.default_rng(0)
    )
    hair_entry = face_hair_metadata(measure_hair_colours(frames, preprocessed.face_landmarks))
    path = exported_path(config)
    write_avatar(
        path,
        avatar_arrays(
            avatar, gaussians, textures, config.training.occlusion_margin, eyes, hair, hair_entry
        ),
    )
    return path


def exported_path(config: Config) -> Path:
    return config.outputs / "export" / "performer.avatar"
