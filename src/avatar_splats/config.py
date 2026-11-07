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


@dataclass(frozen=True)
class Config:
    capture: CaptureConfig
    outputs: Path
    """Directory the stages write their results to."""
    preprocess: PreprocessConfig


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
