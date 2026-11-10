import math

import numpy as np
import torch
from torch.testing import assert_close

from avatar_splats.body_models.genitals import (
    OPEN_SEGMENTS,
    _cross_section,
    _open_map,
    _Opening,
    _polygon,
    _segment_weight,
)


def test_polygon_tells_inside_from_outside_and_measures_the_distance_to_its_edges():
    square = torch.tensor([[0.0, 0], [1, 0], [1, 1], [0, 1]])
    points = torch.tensor([[0.5, 0.5], [0.9, 0.5], [1.5, 0.5], [-0.25, 0.25]])
    inside, distance = _polygon(points, square)
    assert inside.tolist() == [True, True, False, False]
    assert_close(distance, torch.tensor([0.5, 0.1, 0.5, 0.25]))


def test_opening_segments_add_up_to_one_along_the_canal_and_leave_others_alone():
    depth = torch.tensor([0.0, 0.01, 0.033, 0.05, 0.09, 0.1, float("inf")])
    total = sum(_segment_weight(depth, 0.1, k) for k in range(OPEN_SEGMENTS))
    assert_close(total, torch.tensor([1.0, 1, 1, 1, 1, 1, 0]))
    assert _segment_weight(depth, 0.1, 0)[0] == 1  # the entrance opens with the first


def test_opening_moves_its_edge_onto_the_open_circle_and_nothing_beyond_its_reach():
    opening = _Opening("anus", (0.0, 0.0), (0.001, 0.002), 0.01, (0.001, 0.001), 0.1, (0, 0))
    angles = torch.linspace(0, 2 * math.pi, 9)[:-1]
    edge = 1 / ((angles.cos() / 0.001).square() + (angles.sin() / 0.002).square()).sqrt()
    for radius, expected in ((edge, torch.full_like(edge, 0.02)), (0.045, 0.045), (0.05, 0.05)):
        u, s = _open_map(radius * angles.cos(), radius * angles.sin(), opening, 0.02, 0.045, 1.0)
        assert_close((u.square() + s.square()).sqrt(), torch.as_tensor(expected).expand_as(u))


def test_opening_keeps_the_order_of_points_along_a_ray():
    opening = _Opening("anus", (0.0, 0.0), (0.001, 0.001), 0.01, (0.001, 0.001), 0.1, (0, 0))
    radius = torch.linspace(0.001, 0.05, 200)
    u, _ = _open_map(radius, torch.zeros_like(radius), opening, 0.02, 0.045, 1.0)
    assert (u.diff() > 0).all()


def test_cross_section_measures_along_the_surface():
    # A horizontal cylinder of radius 1 along y: the plane y = 0 cuts it in a circle, so a
    # distance along the cut is an angle from the point nearest the centre (its bottom).
    angles = np.linspace(0, 2 * np.pi, 361)[:-1]
    ring = np.stack((np.sin(angles), np.zeros_like(angles), -np.cos(angles)), 1)
    vertices = np.concatenate((ring + [0, -1, 0], ring + [0, 1, 0]))
    n = len(ring)
    faces = np.array(
        [(i, (i + 1) % n, n + i) for i in range(n)]
        + [((i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
    )
    us = np.array([-1.0, 0.0, 0.5, 1.0])
    points = _cross_section(vertices, faces, np.array([0.0, 0, -1.2]), np.array([0, 1.0, 0]), us)
    expected = np.stack((np.sin(us), np.zeros_like(us), -np.cos(us)), 1)
    np.testing.assert_allclose(points, expected, atol=1e-3)
