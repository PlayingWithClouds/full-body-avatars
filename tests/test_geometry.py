import math

import numpy as np
import roma
import torch
from torch.testing import assert_close

from avatar_splats.body_models.hybrid import _umeyama
from avatar_splats.body_models.mesh import read_obj, triangle_frames, uniform_laplacian
from avatar_splats.body_models.skinning import linear_blend_skinning, sparsify_weights
from avatar_splats.cameras import Camera


def grid(n: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    """A flat ``n`` by ``n`` vertex grid in the xy plane, triangulated."""
    xs, ys = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    vertices = torch.stack((xs, ys, torch.zeros_like(xs)), -1).reshape(-1, 3).float()
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = i * n + j, (i + 1) * n + j, (i + 1) * n + j + 1, i * n + j + 1
            faces += [(a, b, c), (a, c, d)]
    return vertices, torch.tensor(faces)


def test_read_obj_triangulates_polygons(tmp_path):
    path = tmp_path / "quad.obj"
    path.write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1/1 2/2 3/3 4/4\n")
    vertices, faces = read_obj(path)
    assert vertices.shape == (4, 3)
    assert faces.tolist() == [[0, 1, 2], [0, 2, 3]]


def test_triangle_frames_are_orthonormal_and_follow_the_normal():
    vertices = torch.tensor([[0.0, 0, 0], [2, 0, 0], [0, 2, 0]])
    frames, size = triangle_frames(vertices, torch.tensor([[0, 1, 2]]))
    assert_close(frames[0].T @ frames[0], torch.eye(3))
    assert_close(frames[0][:, 2], torch.tensor([0.0, 0, 1]))
    assert_close(size, torch.tensor([2.0]))  # sqrt(2 * area)


def test_laplacian_vanishes_on_linear_functions_inside_the_mesh():
    vertices, faces = grid()
    laplacian = uniform_laplacian(faces, len(vertices))
    interior = [5, 6, 9, 10]
    assert_close((laplacian @ vertices)[interior], torch.zeros(4, 3))


def test_skinning_moves_vertices_with_their_bones():
    rest = torch.tensor([[0.0, 0, 0], [1, 0, 0]])
    indices, weights = sparsify_weights(torch.tensor([[1.0, 0], [0.5, 0.5]]), k=2)
    transforms = torch.eye(4).repeat(1, 2, 1, 1)
    transforms[0, 1, :3, 3] = torch.tensor([0, 0, 2.0])
    posed = linear_blend_skinning(rest, indices, weights, transforms)
    assert_close(posed[0], torch.tensor([[0.0, 0, 0], [1, 0, 1]]))


def test_umeyama_recovers_a_similarity():
    points = torch.randn(10, 3, generator=torch.Generator().manual_seed(0))
    rotation = roma.rotvec_to_rotmat(torch.tensor([0.3, -0.2, 1.0]))
    target = 1.7 * points @ rotation.T + torch.tensor([1.0, 2, 3])
    scale, found, translation = _umeyama(points, target)
    assert math.isclose(scale, 1.7, rel_tol=1e-4)
    assert_close(found, rotation, atol=1e-4, rtol=0)
    assert_close(translation, torch.tensor([1.0, 2, 3]), atol=1e-4, rtol=0)


def test_camera_projects_and_unprojects():
    camera = Camera.fixed(200, 400, fov_y_degrees=40)
    point = torch.tensor([0.3, 4.0, -0.5])
    pixel = camera.project(point)
    assert_close(camera.unproject(pixel, depth=4.0), point)
    centre = camera.project(torch.tensor([0.0, 5.0, 0.0]))
    assert_close(centre, torch.tensor([100.0, 200.0]))
    above = camera.project(torch.tensor([0.0, 5.0, 1.0]))
    assert above[1] < 200  # z is up, image y is down


def test_camera_height_and_depth_agree():
    camera = Camera.fixed(200, 400, fov_y_degrees=40)
    depth = camera.depth_for_height(1.6, 300)
    top = camera.project(torch.tensor([0.0, depth, 0.8]))
    bottom = camera.project(torch.tensor([0.0, depth, -0.8]))
    assert np.isclose(float(bottom[1] - top[1]), 300, atol=1e-3)
