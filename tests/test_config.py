from pathlib import Path

import pytest

from avatar_splats.config import load_config

VALID = """
capture:
  body: ../examples/body.mp4
  head_static: ../examples/head.mp4
  head_expressions: /data/expressions.mp4
outputs: ../outputs
preprocess:
  frame_step: 2
  device: cpu
  detector: detector-id
  segmenter: segmenter-id
  pose_estimator: pose-id
  face_landmarker: ../assets/face.task
"""


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "configs" / "test.yaml"
    path.parent.mkdir()
    path.write_text(text)
    return path


def test_relative_paths_resolve_against_config_file(tmp_path):
    config = load_config(write(tmp_path, VALID))
    assert config.capture.body == (tmp_path / "examples" / "body.mp4").resolve()
    assert config.capture.head_expressions == Path("/data/expressions.mp4")
    assert config.preprocess.face_landmarker == (tmp_path / "assets" / "face.task").resolve()
    assert config.preprocess.frame_step == 2


def test_repository_default_config_loads():
    root = Path(__file__).parents[1]
    config = load_config(root / "configs" / "default.yaml")
    assert config.capture.body.parent == root / "examples"
    assert config.outputs == root / "outputs"


def test_unknown_key_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown keys"):
        load_config(write(tmp_path, VALID.replace("outputs:", "output:")))


def test_missing_key_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="head_static"):
        load_config(write(tmp_path, VALID.replace("  head_static: ../examples/head.mp4\n", "")))
