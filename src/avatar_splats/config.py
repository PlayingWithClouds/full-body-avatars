"""Pipeline configuration, loaded from YAML into typed dataclasses.

Relative paths in a config file are resolved against the directory of that file, so a config
means the same thing whether it is used from the repository root or from ``notebooks/``.
"""

from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class CaptureConfig:
    """The three videos of one performer."""

    body: Path
    """Full-body turntable."""
    head_static: Path
    """Head turntable with a neutral face."""
    head_expressions: Path
    """Head with the face moving through expressions."""


VIDEOS = tuple(field.name for field in dataclasses.fields(CaptureConfig))
"""Names of the capture's videos."""


@dataclass(frozen=True)
class PreprocessConfig:
    frame_step: int
    """Process every ``frame_step``-th frame."""
    device: str
    detector: str
    """Hugging Face id of the object detector that finds the performer to segment."""
    segmenter: str
    """Hugging Face id of the SAM2 video model."""
    pose_estimator: str
    """Hugging Face id of the ViTPose model."""
    face_landmarker: Path
    """MediaPipe face landmarker model file; downloaded when missing."""
    hand_landmarker: Path
    """MediaPipe hand landmarker model file; downloaded when missing."""


@dataclass(frozen=True)
class BodyModelsConfig:
    ict_facekit: Path
    """Clone of https://github.com/USC-ICT/ICT-FaceKit."""
    max_bone_influences: int
    """Bones per vertex in the avatar's skinning."""
    neck_seam: float
    """How far below the chin, in metres, the body's neck meets the head model's."""
    neck_blend: float
    """How far up from the seam, in metres, the head model's neck is bent to meet the body's."""


@dataclass(frozen=True)
class FittingConfig:
    device: str
    resolution: int
    """Long side, in pixels, of the images the models are fitted to."""
    fov_y_degrees: float
    """Vertical field of view of the camera, the same for all videos."""
    body_height: float
    """The performer's height in metres. Video alone cannot tell a tall person far away from a
    short one close by, so this sets the avatar's scale."""
    height_weight: float
    eye_distance: float
    """Initial guess of the distance between the pupils in metres, to place the head."""
    surface_samples: int
    """Points spread over a mesh to render its silhouette."""
    silhouette_inset: float
    """How far the silhouette's points are moved inwards, in radii of their blobs, so that the
    blobs' outline is the mesh's (see ``fitting.silhouette``)."""
    iterations: list[int]
    """Iterations of the three fitting phases: placement, shape and pose, local changes."""
    learning_rate: float
    keypoint_weight: float
    hand_weight: float
    """Weight of the hand landmarks, which pose the fingers."""
    silhouette_weight: float
    pose_prior_weight: float
    smoothness_weight: float
    local_change_weight: float
    """Keeps Anny's local changes small: the outline alone should not reshape the body much."""
    identity_prior_weight: float
    expression_sparsity_weight: float


@dataclass(frozen=True)
class TrainingConfig:
    device: str
    resolution: int
    """Long side, in pixels, of the training images."""
    num_gaussians: int
    head_density: float
    """How much more densely the head is covered than the body, per area."""
    hand_density: float
    """How densely the hands are covered, relative to the body: fingers posed a little wrong
    smear Gaussians, the texture less so."""
    coverage_range: list[float]
    """Coverage (the cosine of the most squarely any frame saw a place) below which a place gets
    no Gaussians and above which it gets its full share."""
    occlusion_margin: float
    """How far, in sizes of its triangle, a Gaussian may lie behind the mesh and still show."""
    rebake_every: int
    """Steps between bakes of the textures from the poses training has corrected so far."""
    body_texture_size: int
    head_texture_size: int
    """Side of each of the head atlas's square tiles."""
    sh_degree: int
    iterations: int
    holdout_every: int
    """Every n-th frame is kept out of training to measure quality."""
    ssim_weight: float
    mask_weight: float
    edge_band: int
    """Pixels on either side of the mask's edge whose colours are not compared: there the frames
    mix the performer with the backdrop."""
    colour_weight: float
    """Keeps each Gaussian's colour near the texture's under it, so overlapping Gaussians agree."""
    head_colour_weight: float
    """The same on the head, where Gaussians are dense and overlap most: held loosely, their
    colours drifted from the texture's and the face changed colour as it turned."""
    opacity_weight: float
    """Keeps Gaussians transparent where they add nothing to the textured mesh."""
    offset_weight: float
    """Keeps Gaussians near their triangles."""
    max_offset: float
    """Farthest a Gaussian may move from its anchor, in metres on the rest mesh. Where the head
    videos' neck outline differs from the mesh's, the silhouette loss pulled Gaussians centimetres
    off the skin."""
    max_height: float
    """Farthest a Gaussian may sit above or below its triangle, in metres on the rest mesh.
    Free to move along the normal, they spread into a layer a centimetre thick that looked like
    fur, each frame's edge of the body pulling some outwards."""
    thickness: float
    """Every Gaussian is a disc lying on its triangle, this thick (one standard deviation, in
    metres on the rest mesh): standing up, they looked thin from the training views and streaked
    from others."""
    max_size: float
    """Longest axis a Gaussian may have, in metres (one standard deviation) on the rest mesh.
    A hard limit: as a penalty, a few Gaussians filling a gap in the silhouette grew past it."""
    max_elongation: float
    """How much longer a Gaussian's longest axis may be than its middle one. Training views all
    look from eye level, where a needle pointing at the camera costs nothing; in new poses it
    turns sideways and streaks. Measured across the disc, in the plane of the skin."""
    pose_prior_weight: float
    """Keeps each frame's learned pose and expression corrections small."""
    learning_rates: dict[str, float]


@dataclass(frozen=True)
class Config:
    capture: CaptureConfig
    outputs: Path
    """Directory the stages write their results to."""
    preprocess: PreprocessConfig
    body_models: BodyModelsConfig
    fitting: FittingConfig
    training: TrainingConfig


def load_config(path: str | Path) -> Config:
    """Read a config file, rejecting unknown or missing keys."""
    path = Path(path)
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    return _build(Config, data, base=path.parent.resolve())


def _build[T](cls: type[T], data: dict[str, Any], base: Path) -> T:
    if not isinstance(data, dict):
        raise TypeError(f"{cls.__name__}: expected a mapping, got {type(data).__name__}")

    names = {field.name for field in dataclasses.fields(cls)}
    if unknown := data.keys() - names:
        raise ValueError(f"{cls.__name__}: unknown keys {sorted(unknown)}")

    hints = typing.get_type_hints(cls)
    values = {}
    for name, value in data.items():
        hint = hints[name]
        if dataclasses.is_dataclass(hint):
            values[name] = _build(hint, value, base)
        elif hint is Path:
            values[name] = (base / value).resolve()
        else:
            values[name] = value

    try:
        return cls(**values)
    except TypeError as error:
        raise ValueError(f"{cls.__name__}: {error}") from error
