import json

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from PIL import Image

from avatar_splats.faces import FaceFit, match_expressions, write_faces_json
from avatar_splats.faces.blendshapes import calibrate, neutral_baseline
from avatar_splats.faces.crops import atlas_shape, crop_box, write_crops


def test_match_expressions_orders_by_the_target_and_zeroes_the_missing():
    weights = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], np.float32)

    matched = match_expressions(weights, ["a", "b", "c"], ["c", "genital", "a"])

    assert_allclose(matched, [[0.3, 0.0, 0.1], [0.6, 0.0, 0.4]])


def _fit() -> FaceFit:
    return FaceFit(
        frame_indices=np.array([0, 2]),
        fps=8.0,
        expression_names=np.array(["jawOpen", "eyeBlink_L", "mouthSmile_L"]),
        expressions=np.array([[0.5, 0.0, 0.25], [0.0, 0.0, 0.0]], np.float32),
        rotations=np.array([[0, 0, 0.1], [0, 0, 0.2]], np.float32),
        eye_rotations=np.zeros((2, 2, 3), np.float32),
        found=np.array([True, False]),
        estimate=np.zeros((2, 3), np.float32),
        landmark_error=np.array([3.0, np.nan]),
        estimate_error=np.array([4.0, np.nan]),
        dense_error=np.array([2.0, np.nan]),
        outline=np.zeros((2, 4)),
    )


def test_faces_json_keeps_only_active_expressions_by_name(tmp_path):
    path = tmp_path / "faces" / "v.faces.json"

    write_faces_json(path, "v", _fit(), {"file": "v.faces.jpg", "tile": 8, "columns": 2, "rows": 1})

    document = json.loads(path.read_text())
    assert document["fps"] == 8.0
    first, second = document["frames"]
    assert first["index"] == 0 and first["found"]
    assert first["expressions"] == {"jawOpen": 0.5, "mouthSmile_L": 0.25}
    assert_allclose(first["head"], [0, 0, 0.1])
    assert second["index"] == 2 and not second["found"] and second["expressions"] == {}
    assert len(first["eyes"]) == 2
    assert document["crops"]["file"] == "v.faces.jpg"


def test_face_fit_round_trips_through_npz(tmp_path):
    fit = _fit()
    fit.save(tmp_path / "fit.npz")

    loaded = FaceFit.load(tmp_path / "fit.npz")

    assert_array_equal(loaded.expressions, fit.expressions)
    assert loaded.fps == 8.0


def test_calibration_puts_the_neutral_face_at_zero_and_keeps_the_extremes():
    neutral = np.array([[0.2, 0.0], [0.2, 0.0], [0.2, 0.0], [0.9, 0.5]])
    yaw = np.array([0.0, 0.1, -0.1, 1.2])  # the last frame looks away: it is not neutral

    baseline = neutral_baseline(neutral, yaw, max_yaw=0.35)
    calibrated = calibrate(np.array([[0.2, 0.0], [0.6, 0.0], [1.0, 1.0], [0.1, 0.0]]), baseline)

    assert_allclose(baseline, [0.2, 0.0])
    assert_allclose(calibrated, [[0, 0], [0.5, 0], [1, 1], [0, 0]])


def test_neutral_baseline_needs_a_frontal_face():
    with pytest.raises(ValueError, match="frontal"):
        neutral_baseline(np.full((2, 3), np.nan), np.zeros(2), max_yaw=0.3)


def test_crop_box_is_a_square_around_the_head():
    box = crop_box(np.array([100.0, 50.0, 300.0, 450.0]), margin=1.5)

    assert_allclose(box, [-100.0, -50.0, 500.0, 550.0])


def test_atlas_shape_holds_every_tile():
    assert atlas_shape(81) == (9, 9)
    columns, rows = atlas_shape(10)
    assert columns * rows >= 10


def test_write_crops_pastes_each_frame_in_its_tile(tmp_path):
    frames = [np.full((20, 20, 3), value, np.uint8) for value in (0, 255)]
    boxes = np.array([[0.0, 0, 20, 20], [-10.0, -10, 10, 10]])

    columns, rows = write_crops(tmp_path / "a.jpg", frames, boxes, tile=8, quality=95)

    image = np.asarray(Image.open(tmp_path / "a.jpg"))
    assert (columns, rows) == (2, 1) and image.shape == (8, 16, 3)
    assert image[:, :8].mean() < 10  # first frame: black
    assert image[5:, 13:].mean() > 200 and image[:3, 8:11].mean() < 40  # padding is black
