import numpy as np
from numpy.testing import assert_allclose

from avatar_splats.cameras import body_yaw, head_yaw, unwrap
from avatar_splats.cameras.camera import world_to_camera


def test_unwrap_makes_angles_continuous():
    turn = np.linspace(0, -2 * np.pi, 40)
    wrapped = np.angle(np.exp(1j * turn))
    assert_allclose(unwrap(wrapped), turn, atol=1e-9)


def test_unwrap_fills_gaps_in_the_turning_direction():
    turn = np.linspace(0, -2 * np.pi * 0.95, 60)
    seen = np.angle(np.exp(1j * turn))
    seen[20:50] = np.nan  # the back of the head: no face found
    assert_allclose(unwrap(seen), turn, atol=1e-9)


def body_keypoints(yaw: np.ndarray) -> np.ndarray:
    """COCO keypoints of a stick figure turning on a turntable, seen by the fixed camera."""
    keypoints = np.zeros((len(yaw), 17, 3))
    keypoints[..., 2] = 1
    # Seen from the camera, the left shoulder (+x) is on the image's right when facing it, and
    # the nose (towards -y) moves right as the yaw grows.
    keypoints[:, 5, 0] = 100 * np.cos(yaw)
    keypoints[:, 6, 0] = -100 * np.cos(yaw)
    keypoints[:, 0, 0] = 50 * np.sin(yaw)
    return keypoints


def test_body_yaw_follows_a_full_turn():
    yaw = np.linspace(0, -2 * np.pi, 81)
    assert_allclose(body_yaw(body_keypoints(yaw)), yaw, atol=1e-6)


def test_head_yaw_from_rotating_landmarks():
    rng = np.random.default_rng(0)
    face = rng.normal(size=(50, 3)) * [1, 1, 0.3] + [0, 0, -1]  # facing the camera
    yaw = np.linspace(0, -1.0, 20)
    camera = world_to_camera()[:3, :3].numpy()
    landmarks = []
    for angle in yaw:
        c, s = np.cos(angle), np.sin(angle)
        world = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        landmarks.append(face @ (camera @ world @ camera.T).T)
    assert_allclose(head_yaw(np.array(landmarks)), yaw, atol=1e-6)
