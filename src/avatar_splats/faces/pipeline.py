"""The faces stage: a capture video's face landmarks as the avatar's face, per frame."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from avatar_splats.body_models.hybrid import Avatar
from avatar_splats.body_models.ict import IctFaceKit
from avatar_splats.cameras import head_yaw
from avatar_splats.config import Config
from avatar_splats.data import probe, read_frames
from avatar_splats.faces.blendshapes import calibrate, neutral_baseline
from avatar_splats.faces.convert import FaceFit, convert_faces
from avatar_splats.faces.crops import crop_box, write_crops
from avatar_splats.faces.dense import landmark_vertices
from avatar_splats.fitting.head import HeadFit
from avatar_splats.fitting.pipeline import fitted_path
from avatar_splats.motion.fit import face_expressions
from avatar_splats.preprocess import Preprocessed, preprocessed_path
from avatar_splats.preprocess.face import (
    BLENDSHAPE_NAMES,
    detect_face_blendshapes,
    download_face_landmarker,
)

NEUTRAL_VIDEO = "head_static"
"""The head video of the neutral face: it holds the performer's identity, and her neutral
blendshapes are its frontal frames'."""
HEAD_VIDEOS = ("head_static", "head_expressions")


def convert_video(config: Config, video: str) -> dict[str, Path]:
    """Convert a capture video's face to the avatar's face.

    Needs the preprocessing and the fit of both head videos and the fitted avatar. Writes the
    ``FaceFit``, the same as JSON for the viewer, and the video's crops around the head as a JPEG
    atlas, under ``outputs/faces/``.

    Returns:
        Where each result was written.
    """
    preprocessed = {v: Preprocessed.load(preprocessed_path(config, v)) for v in HEAD_VIDEOS}
    if video not in preprocessed:
        raise ValueError(f"{video} has no face landmarks; convert one of {HEAD_VIDEOS}")
    fits = {v: HeadFit.load(fitted_path(config, v)) for v in HEAD_VIDEOS}
    avatar = Avatar.load(fitted_path(config, "avatar"))
    model = IctFaceKit(config.body_models.ict_facekit)
    faces = config.faces

    # The performer's neutral blendshapes, from the neutral video's frontal frames.
    neutral = preprocessed[NEUTRAL_VIDEO]
    baseline = neutral_baseline(
        _blendshapes(config, NEUTRAL_VIDEO), head_yaw(neutral.face_landmarks), faces.frontal_yaw
    )
    blendshapes = calibrate(_blendshapes(config, video), baseline)
    estimate = face_expressions(
        blendshapes[None], list(BLENDSHAPE_NAMES), list(model.expression_names), faces.smoothing
    )[0]

    mesh_vertices = landmark_vertices(
        model,
        [(fits[v], preprocessed[v].face_landmarks) for v in HEAD_VIDEOS],
        faces.frontal_yaw,
        faces.correspondence_cap,
        config.fitting.device,
    )
    info = probe(getattr(config.capture, video))
    fit = convert_faces(
        preprocessed[video].face_landmarks,
        preprocessed[video].frame_indices,
        (info.width, info.height),
        info.fps / config.preprocess.frame_step,
        model,
        fits[NEUTRAL_VIDEO].identity,
        [str(name) for name in avatar.expression_names],
        estimate,
        mesh_vertices,
        config,
    )
    written = {"faces": faces_path(config, video), "json": faces_json_path(config, video)}
    fit.save(written["faces"])

    boxes = np.stack([crop_box(outline, faces.crop_margin) for outline in fit.outline])
    written["crops"] = faces_crops_path(config, video)
    columns, rows = write_crops(
        written["crops"],
        read_frames(getattr(config.capture, video), step=config.preprocess.frame_step),
        boxes,
        faces.crop_tile,
        faces.crop_quality,
    )
    crops = {
        "file": written["crops"].name,
        "tile": faces.crop_tile,
        "columns": columns,
        "rows": rows,
    }
    write_faces_json(written["json"], video, fit, crops)
    return written


def _blendshapes(config: Config, video: str) -> np.ndarray:
    """``(N, B)`` MediaPipe's blendshapes in every frame of a capture video."""
    frames = list(read_frames(getattr(config.capture, video), step=config.preprocess.frame_step))
    return detect_face_blendshapes(
        frames, download_face_landmarker(config.preprocess.face_landmarker)
    )


def faces_path(config: Config, video: str) -> Path:
    return config.outputs / "faces" / f"{video}.npz"


def faces_crops_path(config: Config, video: str) -> Path:
    """Where the atlas of a video's crops around the head is written."""
    return config.outputs / "faces" / f"{video}.faces.jpg"


def faces_json_path(config: Config, video: str) -> Path:
    """Where the faces of a video are written for the viewer."""
    return config.outputs / "faces" / f"{video}.faces.json"


def write_faces_json(path: Path, video: str, fit: FaceFit, crops: dict | None = None) -> None:
    """Write the faces as JSON: per frame, the avatar's expression weights (only those that
    are not zero, by name), the head's and the eyes' rotation vectors in world axes, and the
    frame's index in the video, for the viewer's grid of faces. ``crops`` says where the video's
    crops are: their atlas (a file next to this one), its tile size in pixels and its columns and
    rows, row by row in the order of the frames."""
    frames = []
    for i in range(len(fit.frame_indices)):
        weights = {
            str(name): round(float(w), 4)
            for name, w in zip(fit.expression_names, fit.expressions[i], strict=True)
            if w > 1e-3
        }
        frames.append(
            {
                "index": int(fit.frame_indices[i]),
                "found": bool(fit.found[i]),
                "expressions": weights,
                "head": [round(float(v), 5) for v in fit.rotations[i]],
                "eyes": [[round(float(v), 5) for v in eye] for eye in fit.eye_rotations[i]],
            }
        )
    document = {"format_version": 1, "video": video, "fps": float(fit.fps), "frames": frames}
    if crops:
        document["crops"] = crops
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, separators=(",", ":")))
