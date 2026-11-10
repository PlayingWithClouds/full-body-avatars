import numpy as np
import pytest

from avatar_splats.preprocess.identity import (
    TEMPLATE,
    alignment,
    keypoint_face_points,
    landmark_face_points,
    main_identity,
)
from avatar_splats.preprocess.quality import (
    A_POSE,
    BODY_JOINTS,
    MIRRORED,
    FaceQualitySettings,
    QualitySettings,
    a_pose_distance,
    at_edge,
    body_mask,
    face_outline,
    lighting,
    measure_faces,
    measure_frames,
    occlusion,
    score_faces,
    score_frames,
)


def person(body: np.ndarray, scale: float = 200.0, shift=(400.0, 300.0)) -> np.ndarray:
    """``(17, 3)`` COCO keypoints with ``(12, 2)`` body joints in torso lengths, the face above
    the shoulders, all confident."""
    points = np.zeros((17, 3))
    points[:, 2] = 0.9
    points[list(BODY_JOINTS), :2] = body * scale + shift
    points[:5, :2] = (
        np.array([(0, -1.45), (0.07, -1.5), (-0.07, -1.5), (0.15, -1.45), (-0.15, -1.45)]) * scale
        + shift
    )
    return points


def test_a_pose_distance_is_zero_for_any_size_place_and_side():
    assert a_pose_distance(person(A_POSE), 0.3) == pytest.approx(0, abs=1e-9)
    assert a_pose_distance(person(A_POSE, 37.0, (5.0, 900.0)), 0.3) == pytest.approx(0, abs=1e-9)
    from_behind = person(A_POSE[list(MIRRORED)])
    assert a_pose_distance(from_behind, 0.3) == pytest.approx(0, abs=1e-9)


def test_a_pose_distance_grows_with_raised_arms_and_needs_joints():
    raised = A_POSE.copy()
    raised[[4, 5], 1] = -2.0  # wrists above the head
    assert a_pose_distance(person(raised), 0.3) > 0.1
    missing = person(A_POSE)
    missing[[9, 10, 15], 2] = 0.1
    assert np.isnan(a_pose_distance(missing, 0.3))


def test_occlusion_sees_hands_on_the_body():
    assert occlusion(person(A_POSE), 0.3) == pytest.approx(0, abs=0.01)
    covering = A_POSE.copy()
    covering[[2, 3]] = [(0.45, -0.5), (-0.45, -0.5)]  # elbows at the sides
    covering[[4, 5]] = [(0.0, -0.4), (0.05, -0.3)]  # hands on the belly
    assert occlusion(person(covering), 0.3) > 0.05


def test_body_mask_covers_the_torso():
    points = person(A_POSE)
    mask = body_mask(points, (800, 800), 0.3)
    middle = np.round(points[[5, 6, 11, 12], :2].mean(0)).astype(int)
    assert mask[middle[1], middle[0]]
    assert not mask[5, 5]


def test_lighting_sees_dark_and_shadowed():
    mask = np.ones((40, 40), bool)
    even = np.full((40, 40, 3), 180, np.uint8)
    shadowed = even.copy()
    shadowed[:, :20] = 40
    brightness, clipped, evenness = lighting(even, mask)
    assert brightness == pytest.approx(180 / 255, abs=0.01) and clipped == 0
    assert evenness == pytest.approx(1)
    assert lighting(shadowed, mask)[2] < 0.3
    assert np.isnan(lighting(even, ~mask)[0])
    spotted = even.copy()
    spotted[::2] = 30  # a print: thin dark stripes over half the body
    assert lighting(spotted, mask)[2] < 0.3
    assert lighting(spotted, mask, blur=4.0)[2] > 0.9
    assert lighting(shadowed, mask, blur=4.0)[2] < 0.4


def test_at_edge():
    assert at_edge(np.array([0.0, 50, 100, 150]), (200, 300), 0.01)
    assert not at_edge(np.array([10.0, 50, 100, 150]), (200, 300), 0.01)


def settings(**changes) -> QualitySettings:
    values = dict(
        min_confidence=0.3, edge=0.01, good_pixels=1000.0, good_focus=0.05, dark=0.2,
        bright=0.5, max_clipped=0.2, uneven=0.3, even=0.7, pose_tolerance=0.6,
        max_occlusion=0.3, max_overlap=0.3, other_face=0.2, same_face=0.4,
        weights={"resolution": 2.0, "sharpness": 2.0, "pose": 1.0},
        gates={"whole": 1.0, "resolution": 0.5, "pose": 0.2}, min_score=0.5,
    )  # fmt: skip
    values.update(changes)
    return QualitySettings(**values)


def test_score_prefers_detail_over_pose_and_says_why():
    measures = {
        "pixels": np.array([1000.0, 700.0, 1000.0, 300.0]),
        "focus": np.array([0.05, 0.05, 0.05, 0.05]),
        "brightness": np.full(4, 0.5), "clipped": np.zeros(4), "evenness": np.full(4, 0.7),
        "pose": np.array([0.2, 0.0, 0.2, 0.0]),  # the second's pose is better
        "occlusion": np.zeros(4), "joints": np.array([12.0, 12, 11, 12]),
        "cut": np.zeros(4, bool), "overlap": np.zeros(4),
    }  # fmt: skip
    factors, score, passed, reasons = score_frames(measures, settings())
    assert score[0] > score[1]  # sharper detail beats the better pose
    assert passed[0] and passed[1] and reasons[0] == ""
    assert not passed[2] and reasons[2].startswith("whole")
    assert not passed[3] and "resolution 0.30" in reasons[3]
    assert set(factors) >= {"whole", "pose", "alone"}


def test_measure_frames_on_a_drawn_person():
    points = person(A_POSE, 150.0, (300.0, 250.0))
    frame = np.full((600, 600, 3), 30, np.uint8)
    frame[body_mask(points, (600, 600), 0.3)] = 200
    box = np.array([[100.0, 20, 500, 560]])
    measures = measure_frames([frame], box, points[None], 0.5, np.zeros(1), settings())
    assert measures["pixels"][0] == pytest.approx(1080)
    assert measures["pose"][0] == pytest.approx(0, abs=1e-6)
    assert measures["brightness"][0] == pytest.approx(200 / 255, abs=0.01)
    assert measures["joints"][0] == 12 and not measures["cut"][0]


def test_identity_gates_other_people_but_not_unseen_faces():
    measures = {
        "pixels": np.full(3, 1000.0), "focus": np.full(3, 0.05), "brightness": np.full(3, 0.5),
        "clipped": np.zeros(3), "evenness": np.full(3, 0.7), "pose": np.zeros(3),
        "occlusion": np.zeros(3), "joints": np.full(3, 12.0), "cut": np.zeros(3, bool),
        "overlap": np.zeros(3), "identity": np.array([0.6, 0.05, np.nan]),
    }  # fmt: skip
    _, _, passed, reasons = score_frames(measures, settings(gates={"identity": 0.5}))
    assert passed[0] and passed[2]  # the actor; her back
    assert not passed[1] and reasons[1].startswith("identity")


def face_settings() -> FaceQualitySettings:
    return FaceQualitySettings(
        min_confidence=0.3, edge=0.01, good_pixels=300.0, good_focus=0.05, dark=0.2, bright=0.5,
        max_clipped=0.2, uneven=0.3, even=0.7, max_occlusion=0.3, max_overlap=0.3,
        other_face=0.2, same_face=0.4, weights={"resolution": 1.0, "occlusion": 1.0},
        gates={"found": 1.0, "occlusion": 0.5}, min_score=0.3,
    )  # fmt: skip


def test_faces_and_hands_over_them():
    points = person(A_POSE, 150.0, (300.0, 300.0))
    frame = np.full((600, 600, 3), 160, np.uint8)
    outline = face_outline(points, None, 0.3)
    assert outline is not None
    covered = points.copy()
    covered[9, :2] = outline.mean(0)  # a hand on the face
    covered[7, :2] = outline.mean(0) + (0, 60)
    measures = measure_faces(
        [frame, frame], np.stack([points, covered]), None, None, 1.0, np.zeros(2), face_settings()
    )
    assert measures["found"].all() and measures["pixels"][0] > 50
    assert measures["occlusion"][0] == 0 and measures["occlusion"][1] > 0.2
    _, _, passed, reasons = score_faces(measures, face_settings())
    assert passed[0] and not passed[1] and "occlusion" in reasons[1]
    nobody = np.full((1, 17, 3), np.nan)
    gone = measure_faces([frame], nobody, None, None, 1.0, np.zeros(1), face_settings())
    assert not gone["found"][0]
    assert score_faces(gone, face_settings())[3][0].startswith("found")


def test_alignment_undoes_a_roll():
    angle = np.radians(20)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    points = TEMPLATE @ rotation.T * 2 + (40, 60)
    matrix, roll = alignment(points)
    assert roll == pytest.approx(20, abs=0.1)
    back = points @ matrix[:, :2].T + matrix[:, 2]
    assert np.abs(back - TEMPLATE).max() < 1e-3


def test_face_points_left_eye_first():
    keypoints = np.zeros((17, 3))
    keypoints[:3] = [(50, 60, 0.9), (60, 50, 0.9), (40, 50, 0.9)]  # nose, her left, right eye
    assert keypoint_face_points(keypoints, 0.3).tolist() == [[40, 50], [60, 50], [50, 60]]
    keypoints[1, 2] = 0.1
    assert keypoint_face_points(keypoints, 0.3) is None
    assert landmark_face_points(np.full((478, 3), np.nan)) is None


def test_main_identity_is_the_most_seen_face():
    rng = np.random.default_rng(3)
    actor, other = rng.normal(size=512), rng.normal(size=512)

    def faces(centre, count):
        noisy = centre + 0.6 * rng.normal(size=(count, 512)) * np.linalg.norm(centre) / 22
        return noisy / np.linalg.norm(noisy, axis=1, keepdims=True)

    embeddings = np.vstack([faces(actor, 30), faces(other, 10), np.full((5, 512), np.nan)])
    found, members = main_identity(embeddings, same=0.4)
    assert found @ (actor / np.linalg.norm(actor)) > 0.8
    assert set(members) <= set(range(30)) and len(members) > 20
