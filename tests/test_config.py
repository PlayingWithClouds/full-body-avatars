from pathlib import Path

import pytest
import yaml

from avatar_splats.config import load_config

DEFAULT = Path(__file__).parents[1] / "configs" / "default.yaml"


def write(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "configs" / "test.yaml"
    path.parent.mkdir()
    path.write_text(yaml.safe_dump(data))
    return path


@pytest.fixture
def data() -> dict:
    return yaml.safe_load(DEFAULT.read_text())


def test_relative_paths_resolve_against_config_file(tmp_path, data):
    data["capture"]["body"] = "../examples/body.mp4"
    data["capture"]["head_expressions"] = "/data/expressions.mp4"
    config = load_config(write(tmp_path, data))
    assert config.capture.body == (tmp_path / "examples" / "body.mp4").resolve()
    assert config.capture.head_expressions == Path("/data/expressions.mp4")
    assert config.body_models.ict_facekit == (tmp_path / "assets" / "ict-facekit").resolve()


def test_repository_default_config_loads():
    config = load_config(DEFAULT)
    root = DEFAULT.parents[1]
    assert config.capture.body.parent == root / "examples"
    assert config.outputs == root / "outputs"
    assert len(config.fitting.iterations) == 3


def test_unknown_key_is_rejected(tmp_path, data):
    data["fitting"]["learning_rates"] = 0.1
    with pytest.raises(ValueError, match="unknown keys"):
        load_config(write(tmp_path, data))


def test_missing_key_is_rejected(tmp_path, data):
    del data["capture"]["head_static"]
    with pytest.raises(ValueError, match="head_static"):
        load_config(write(tmp_path, data))
