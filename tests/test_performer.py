import numpy as np
import pytest

from avatar_splats.fitting.performer import (
    IRISES,
    eu_shoe_size,
    foot_length,
    frontal_scores,
    iris_diameters,
    pupil_distances,
    scale_from_size,
    scaled_measurements,
)


def standing(facing: float = 1.0) -> np.ndarray:
    """``(17, 3)`` COCO keypoints of someone upright, 100 px from shoulders to hips, facing the
    camera (``facing`` 1) or turned (shoulders and hips ``facing`` as wide)."""
    points = np.zeros((17, 3))
    points[:, 2] = 0.9
    points[:5, :2] = (200, 60)
    for left, right, y, half in ((5, 6, 100, 40), (11, 12, 200, 30)):
        points[left, :2] = (200 + facing * half, y)
        points[right, :2] = (200 - facing * half, y)
    for k in range(7, 11):
        points[k, :2] = (200 + (40 if k % 2 else -40), 150)
    for k in range(13, 17):
        points[k, :2] = (200 + (25 if k % 2 else -25), 300 + 50 * (k > 14))
    return points


def test_frontal_scores_prefer_facing_the_camera():
    turned = standing(0.4)
    back = standing(-1.0)  # her left on the image's left
    missing = standing()
    missing[15, 2] = 0.1
    scores = frontal_scores(np.stack([standing(), turned, back, missing]), 0.3)
    assert scores[0] == pytest.approx(1.0)
    assert 0 < scores[1] < 0.5
    assert scores[2] == 0
    assert scores[3] == 0


def test_frontal_scores_want_her_upright():
    lying = standing()[:, [1, 0, 2]]  # x and y swapped: the torso across the image
    assert frontal_scores(lying[None], 0.3)[0] == 0


def eyes(diameter: float, apart: float, depth: float = 0.0) -> np.ndarray:
    """``(1, 478, 3)`` landmarks with two irises ``diameter`` pixels wide, ``apart`` pixels
    apart in 3D, the right one ``depth`` pixels further."""
    landmarks = np.zeros((1, 478, 3))
    offset = np.sqrt(apart**2 - depth**2)
    for (centre, *rim), x, z in zip(IRISES, (0.0, offset), (0.0, depth), strict=True):
        landmarks[0, centre] = (x, 0, z)
        for angle, k in zip((0, 90, 180, 270), rim, strict=True):
            a = np.radians(angle)
            landmarks[0, k] = (x + diameter / 2 * np.cos(a), diameter / 2 * np.sin(a), z)
    return landmarks


def test_iris_and_pupils_in_pixels():
    landmarks = eyes(diameter=12.0, apart=60.0, depth=20.0)
    assert iris_diameters(landmarks)[0] == pytest.approx(12.0)
    assert pupil_distances(landmarks)[0] == pytest.approx(60.0)


def test_iris_diameter_keeps_the_longer_rim():
    landmarks = eyes(diameter=12.0, apart=60.0)
    landmarks[0, :, 0] *= 0.5  # the head turned: across shortened, down not
    assert iris_diameters(landmarks)[0] == pytest.approx(12.0)


def test_scale_from_size():
    # 10 px at 2 m through a 1000 px lens is 2 cm: a 4 cm thing is twice as large
    assert scale_from_size(10.0, 2.0, 1000.0, 0.04) == pytest.approx(2.0)


def test_shoe_size_round_trip():
    assert eu_shoe_size(0.230) == pytest.approx(36.75)
    assert foot_length(eu_shoe_size(0.245)) == pytest.approx(0.245)


def test_scaled_measurements():
    scaled = scaled_measurements({"weight": 50.0, "waist": 0.6, "foot": 0.23}, 1.1)
    assert scaled["weight"] == pytest.approx(50.0 * 1.1**3)
    assert scaled["waist"] == pytest.approx(0.66)
    assert scaled["foot"] == pytest.approx(0.253)
