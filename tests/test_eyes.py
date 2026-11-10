import numpy as np
from numpy.testing import assert_allclose
from scipy.spatial import ConvexHull

from avatar_splats.body_models.hybrid import BODY, EYES, Avatar
from avatar_splats.export.eyes import EYE_LANDMARKS, NOSE_TIP, EyeLook, eye_metadata, measure_eyes

IRIS = np.array([60, 110, 90])
SCLERA = np.array([235, 230, 225])
SKIN = np.array([200, 160, 140])


def face_frame(radius: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """A frame with two eyes, 64 px wide, and the MediaPipe landmarks of a frontal face."""
    frame = np.broadcast_to(SKIN, (200, 300, 3)).astype(np.uint8).copy()
    landmarks = np.zeros((478, 3))
    ys, xs = np.mgrid[:200, :300]
    for (centre, ring, corners, lids), middle in zip(
        EYE_LANDMARKS, ((90, 100), (210, 100)), strict=True
    ):
        x, y = middle
        eye = ((xs - x) / 32) ** 2 + ((ys - y) / 14) ** 2 < 1
        frame[eye] = SCLERA
        distance = np.hypot(xs - x, ys - y)
        frame[distance < radius] = IRIS
        frame[distance < 0.4 * radius] = 0  # the pupil
        landmarks[centre, :2] = middle
        for index, angle in zip(ring, np.arange(4) * np.pi / 2, strict=True):
            landmarks[index, :2] = x + radius * np.cos(angle), y + radius * np.sin(angle)
        landmarks[list(corners), :2] = [[x - 32, y], [x + 32, y]]
        landmarks[list(lids), :2] = [[x, y - 14], [x, y + 14]]
    landmarks[NOSE_TIP, :2] = 150, 140
    return frame, landmarks


def test_eyes_are_measured_in_frontal_frames_with_open_eyes():
    frame, landmarks = face_frame()
    closed = landmarks.copy()
    for _, _, _, lids in EYE_LANDMARKS:
        closed[list(lids), 1] = 100
    turned = landmarks.copy()
    turned[NOSE_TIP, 0] = 190
    frames = np.stack([frame, np.zeros_like(frame), np.zeros_like(frame)])
    look = measure_eyes(frames, np.stack([landmarks, closed, turned]))
    assert_allclose(look.iris_colour, IRIS / 255)
    assert_allclose(look.sclera_colour, SCLERA / 255)
    assert_allclose(look.iris_width, 20 / 64)


def test_eyeballs_are_found_from_their_shape():
    directions = np.random.default_rng(0).normal(size=(400, 3))
    sphere = 0.012 * directions / np.linalg.norm(directions, axis=1, keepdims=True)
    faces = ConvexHull(sphere).simplices
    centres = np.array([[0.03, -0.1, 1.6], [-0.03, -0.1, 1.6]])
    corners = np.array([[0.015, -0.11, 1.6], [0.045, -0.11, 1.6]])
    vertices = np.concatenate([sphere + centres[0], sphere + centres[1], corners, -corners])
    triangles = np.concatenate([faces, faces + len(sphere), [[800, 801, 802]]])
    landmarks = np.zeros(68, int)
    landmarks[[36, 39, 42, 45]] = 800, 801, 802, 803
    avatar = Avatar(
        rest_vertices=vertices.astype(np.float32),
        faces=triangles,
        face_parts=np.array([EYES] * (len(triangles) - 1) + [BODY], np.int8),
        uvs=np.zeros((len(triangles), 3, 2), np.float32),
        face_textures=np.zeros(len(triangles), np.int8),
        bone_indices=np.zeros((len(vertices), 1), int),
        bone_weights=np.ones((len(vertices), 1), np.float32),
        bone_labels=np.array(["root"]),
        bone_parents=np.array([-1]),
        bind_poses=np.eye(4, dtype=np.float32)[None],
        head_start=0,
        expression_names=np.array([]),
        expression_basis=np.zeros((0, len(vertices), 3), np.float32),
    )
    look = EyeLook(np.full(3, 0.4), np.full(3, 0.9), iris_width=0.4)
    eyes = eye_metadata(avatar, look, landmarks.tolist())

    assert [eyeball["side"] for eyeball in eyes["eyeballs"]] == ["left", "right"]
    for side, (eyeball, centre) in enumerate(zip(eyes["eyeballs"], centres, strict=True)):
        assert_allclose(eyeball["centre"], centre, atol=1e-6)
        assert_allclose(eyeball["radius"], 0.012, rtol=1e-4)
        own = vertices[400 * side : 400 * (side + 1)]
        assert vertices[eyeball["apex"], 1] == own[:, 1].min()  # the front, towards -y
        assert vertices[eyeball["top"], 2] == own[:, 2].max()
    # The iris spans 0.4 of an eye 3 cm wide: 6 mm across, on an eyeball of 12 mm radius.
    assert_allclose(eyes["iris_angle"], np.arcsin(0.006 / 0.012), rtol=1e-4)
