import numpy as np
import torch
from torch.testing import assert_close

from avatar_splats.body_models.mesh import boundary_loops, closest_points, zip_loops


def ring(count: int, radius: float, height: float, phase: float = 0.0) -> np.ndarray:
    angle = phase + np.linspace(0, 2 * np.pi, count, endpoint=False)
    return np.stack((radius * np.cos(angle), radius * np.sin(angle), np.full(count, height)), 1)


def test_boundary_loops_of_a_strip_are_its_two_rims():
    # A closed band of quads between two rings of 6 vertices.
    n = 6
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces += [(i, j, n + j), (i, n + j, n + i)]
    loops = boundary_loops(np.array(faces))
    assert sorted(sorted(loop.tolist()) for loop in loops) == [list(range(6)), list(range(6, 12))]


def test_zipped_loops_close_the_gap_facing_outwards():
    lower, upper = ring(9, 1.0, 0.0), ring(5, 0.8, 0.3, phase=0.4)[::-1]
    triangles = zip_loops(lower, upper)
    points = np.concatenate((lower, upper))
    assert len(triangles) == 14
    # Every vertex used, and the band has exactly the two loops as its boundary.
    assert set(triangles.ravel()) == set(range(14))
    loops = boundary_loops(triangles)
    assert sorted(len(loop) for loop in loops) == [5, 9]
    a, b, c = (points[triangles[:, k]] for k in range(3))
    normals = np.cross(b - a, c - a)
    outward = (a + b + c) / 3
    outward[:, 2] = 0
    assert ((normals * outward).sum(1) > 0).all()


def test_closest_points_on_a_triangle():
    vertices = torch.tensor([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    faces = torch.tensor([[0, 1, 2]])
    points = torch.tensor([[0.2, 0.2, 0.5], [2.0, -1, 0], [-1, 0.5, 0.1]])
    face, weights = closest_points(points, vertices, faces)
    nearest = (weights[..., None] * vertices[faces[face]]).sum(1)
    assert_close(nearest, torch.tensor([[0.2, 0.2, 0], [1, 0, 0], [0, 0.5, 0]]))
