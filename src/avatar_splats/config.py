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
class GenitalsConfig:
    resolution: float
    """Spacing, in metres, of the generated surface's vertices."""
    vagina_depth: float
    """Length, in metres, of the vaginal canal."""
    anus_depth: float
    """Length, in metres, of the anal canal."""
    open_radius: float
    """Radius, in metres, of an opening and its canal fully opened."""
    sculpt: Path
    """STL of a sculpted vulva, a relief seen from behind (its z towards the viewer), whose
    surface detail the generated vulva takes."""
    sculpt_midline: float
    """x of the sculpt's midline, in its own units."""
    sculpt_back: float
    """y of the back end of the sculpt's vulva, where the labia meet behind the entrance."""
    sculpt_front: float
    """y of its front end, where the sculpt stops: from there on, the hood is generated."""
    sculpt_introitus: float
    """y of the middle of its entrance."""


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
    subdivisions: int
    """Times the joined mesh is subdivided (each splitting every triangle in four and smoothing
    the surface), so it can carry detail."""
    genitals: GenitalsConfig


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
    gaze_prior_weight: float
    """Keeps the eyes looking ahead where the irises say little: in profile, or closed."""


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
    align_rounds: int
    """Rounds of baking the textures and aligning the frames to them by their pixels."""
    align_steps: int
    """Steps of aligning per round, one frame each."""
    align_resolution: int
    """Long side, in pixels, of the frames aligned; the bakes use the full frames."""
    align_silhouette_weight: float
    """Weight of the mesh's outline against the performer's mask while aligning."""
    rebake_every: int
    """Steps between bakes of the textures from the poses training has corrected so far."""
    body_texture_size: int
    head_texture_size: int
    """Side of each of the head atlas's square tiles."""
    fallback_videos: list[str]
    """Videos that colour a texel only where no other video sees it: their faces change too much
    to be averaged into one texture (wrinkles, brows), but only they see inside the mouth."""
    detail_scale: float
    """Standard deviation, in pixels of the frames, of the blur splitting each frame into broad
    colour, averaged over many frames, and detail, taken from one frame per patch: features
    smaller than a few times this come out sharp; larger, the patches' seams show more."""
    seam_weight: float
    """How much a triangle prefers to take its fine detail from a frame one of its neighbours
    takes it from, in shares of the quality of its own best frame: higher, fewer and larger
    patches, a little less sharp."""
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
class MotionConfig:
    """Extracting motion from videos of other people, to play on the avatar. The detector,
    segmenter, pose estimator and landmarkers are the preprocess stage's."""

    device: str
    frame_step: int
    """Process every ``frame_step``-th frame."""
    max_people: int
    """People followed through a video, the most confidently detected first."""
    detection_threshold: float
    """Confidence a detection needs to count as a person."""
    detection_every: int
    """Frames between those searched for the one where everyone is seen, to prompt SAM2."""
    fov_y_degrees: float
    """Vertical field of view of the motion videos' camera; a guess, not fitted."""
    track_frames: int
    """Frames read and tracked at once; long videos are tracked a stretch at a time."""
    resolution: int
    """Long side, in pixels, of the masks and of the images the bodies are fitted to."""
    window_frames: int
    """Frames fitted together; long videos are fitted in overlapping windows."""
    window_overlap: int
    """Frames each window shares with the one before, blended across."""
    iterations: list[int]
    """Most steps of the three fitting phases: keypoints, then hands, then silhouettes."""
    check_every: int
    """Steps between checks of whether a phase's loss still falls."""
    tolerance: float
    """A phase stops when its loss fell by less than this share since the last check."""
    learning_rate: float
    chunk_frames: int
    """Frames rendered at once for the silhouettes; gradients add up over the chunks."""
    surface_samples: int
    keypoint_weight: float
    hand_weight: float
    silhouette_weight: float
    occluded_keypoint_weight: float
    """Share of its weight a keypoint keeps where it lies on someone else and not on its own
    person: either hidden behind them or mistaken for one of theirs."""
    smoothness_weight: float
    pose_prior_weight: float
    normal_weight: float
    """Per 1 - cosine between the fitted body's normals and estimated ones, where the
    observations carry them (``Observations.normals``): which way limbs turn."""
    depth_weight: float
    """Per metre between the fitted body's depth and estimated depths, scaled to agree in
    their median, where the observations carry them: which part is in front."""
    keypoint_depth_weight: float
    """Per metre between the fitted keypoints' depths and the estimated ones, scaled to agree
    in their mean, where the observations carry them (``Observations.keypoint_depths``): which
    joint is in front, also where the surface between them is hidden."""
    pose_fingers: bool
    """Whether the fingers are posed per frame; if not they are held open and relaxed
    (``fitting.body.relaxed_hands``) and the hand landmarks only turn the hands. Hands held
    open through a video are better off without: per frame, small and blurred hands bend the
    fingers wrongly."""
    finger_prior_weight: float
    """Per squared radian the posed fingers turn from an open, relaxed hand: unsure hand points
    leave them so."""
    joint_limit_weight: float
    """Per squared radian a bone turns past its limit (``motion.priors.JOINT_LIMITS``): the
    twist bones only twist, elbows and knees are hinges."""
    knee_weight: float
    """Keeps knees from bending forwards, the most common mirror ambiguity of one camera."""
    contact_weight: float
    """Keeps the bodies of different people from passing through each other."""
    self_contact_weight: float
    """Keeps a body's limbs from passing through its own body (``motion.priors.self_overlap``):
    one body has many more pairs of spheres than two touching, so this weighs less."""
    contact_slack: float
    """Share of the bodies' radii they may sink into each other: skin gives way at contact."""
    expression_smoothing: int
    """Frames in the median filter over the face's blendshape weights."""


@dataclass(frozen=True)
class FacesConfig:
    """Converting a video's face landmarks to the avatar's face. The camera, the weights and the
    priors are the fitting stage's."""

    iterations: list[int]
    """Steps of the two fitting phases: the head's pose with the estimated expressions, then its
    pose, expressions and gaze."""
    learning_rate: float
    prior_weight: float
    """Holds the expressions near the blendshape estimate, against the landmarks."""
    smoothing: int
    """Frames in the median filter over the blendshape weights."""
    frontal_yaw: float
    """How far from frontal, in radians, a frame of the neutral head video still gives the
    performer's neutral blendshapes, and the landmarks' vertices on the head model."""
    correspondence_cap: float
    """A mesh landmark's distance to a vertex counts at most this, in image heights, when its
    vertex is chosen."""
    crop_margin: float
    """Side of a frame's crop of the head, in sides of the head's outline."""
    crop_tile: int
    """Side in pixels of a crop in the viewer's atlas."""
    crop_quality: int
    """JPEG quality of the atlas."""


MEASUREMENT_NAMES = ("weight", "bust", "underbust", "waist", "hips", "foot")
"""What ``PerformerConfig.measurements`` may give (``body_models.measurements``)."""


@dataclass(frozen=True)
class PerformerConfig:
    """What is known of the performer, which the body fits keep to rather than guess."""

    measurements: dict[str, float]
    """Her measurements, which the body is shaped to before any frame is seen and held to
    through the fit: weight in kg, the circumferences (bust, underbust, waist, hips) and the
    feet's length, heel to big toe (EU shoe size / 1.5 - 1.5 cm), in metres;
    her height is ``FittingConfig.body_height``. Empty where none are known."""
    phenotype: dict[str, float]
    """Anny phenotype values held through the fit (gender 1 is female, 0 male), for what no
    measurement says, such as how the breasts look (cupsize, firmness), chosen by eye."""

    def __post_init__(self):
        if unknown := self.measurements.keys() - set(MEASUREMENT_NAMES):
            raise ValueError(f"PerformerConfig: unknown measurements {sorted(unknown)}")


@dataclass(frozen=True)
class Config:
    capture: CaptureConfig
    outputs: Path
    """Directory the stages write their results to."""
    performer: PerformerConfig
    preprocess: PreprocessConfig
    body_models: BodyModelsConfig
    fitting: FittingConfig
    training: TrainingConfig
    motion: MotionConfig
    faces: FacesConfig


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
