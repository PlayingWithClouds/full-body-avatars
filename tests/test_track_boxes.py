import numpy as np

from avatar_splats.motion.observe import box_iou, track_boxes


def _box(x: float, y: float = 0.0, size: float = 10.0) -> list[float]:
    return [x, y, x + size, y + size, 0.9]


def test_box_iou():
    a = np.array([[0, 0, 10, 10]], float)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], float)
    np.testing.assert_allclose(box_iou(a, b), [[1, 50 / 150, 0]])


def test_track_boxes_follows_people_forwards_and_backwards():
    # Two people walking apart, detected in either order.
    detections = [
        np.array([_box(50 - 2 * t), _box(60 + 2 * t)][:: 1 - 2 * (t % 2)]) for t in range(6)
    ]
    start = 3
    boxes = track_boxes(detections, start, np.array([_box(44)[:4], _box(66)[:4]]), 0.2)
    np.testing.assert_allclose(boxes[0, :, 0], [50 - 2 * t for t in range(6)])
    np.testing.assert_allclose(boxes[1, :, 0], [60 + 2 * t for t in range(6)])


def test_track_boxes_leaves_missed_frames_nan_and_finds_people_again():
    detections = [np.array([_box(0)]), np.zeros((0, 5)), np.array([_box(100)]), np.array([_box(1)])]
    boxes = track_boxes(detections, 0, np.array([_box(0)[:4]]), 0.2)
    assert not np.isnan(boxes[0, 0]).any()
    # Nothing on frame 1; someone far away on frame 2 is not them; frame 3 is.
    assert np.isnan(boxes[0, 1]).all() and np.isnan(boxes[0, 2]).all()
    np.testing.assert_allclose(boxes[0, 3], _box(1)[:4])


def test_track_boxes_ignores_people_without_a_box():
    detections = [np.array([_box(0)]), np.array([_box(0)])]
    start = np.array([[np.nan] * 4, _box(0)[:4]])
    boxes = track_boxes(detections, 0, start, 0.2)
    assert np.isnan(boxes[0, 1]).all()
    np.testing.assert_allclose(boxes[1, 1], _box(0)[:4])
