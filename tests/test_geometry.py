import math

import numpy as np
import roma
import torch
from torch.testing import assert_close

from avatar_splats.body_models import ict
from avatar_splats.body_models.hybrid import _umeyama
from avatar_splats.body_models.mesh import (
    adjacent_faces,
    loop_subdivision,
    read_obj,
    read_stl,
    triangle_frames,
    uniform_laplacian,
)
from avatar_splats.body_models.skinning import (
    linear_blend_skinning,
    skinned_point_weights,
    skinned_points,
    sparsify_weights,
)
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


def test_loop_subdivision_keeps_a_plane_flat_and_splits_every_triangle():
    vertices, faces = grid(4)
    split, matrix = loop_subdivision(faces, len(vertices))
    assert len(split) == 4 * len(faces)
    new = matrix @ vertices
    assert_close(new[:, 2], torch.zeros(len(new)))
    assert_close(torch.sparse.sum(matrix, dim=1).to_dense(), torch.ones(len(new)))
    # Every new triangle faces the way the old ones did.
    corners = new[split]
    normals = torch.linalg.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    old = vertices[faces[0]]
    facing = torch.linalg.cross(old[1] - old[0], old[2] - old[0])[2]
    assert (normals[:, 2] * facing > 0).all()


def test_loop_subdivision_follows_its_rules_on_a_tetrahedron():
    vertices = torch.tensor([[1.0, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])
    faces = torch.tensor([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]])
    split, matrix = loop_subdivision(faces, 4)
    new = matrix @ vertices
    assert len(new) == 10 and len(split) == 16
    # Corners: 7/16 of themselves and 3/16 of each of three neighbours; edges: 3/8 of each end
    # and 1/8 of each opposite corner. Rounder, and much smaller, as Loop makes a tetrahedron.
    assert_close(new[:4], vertices / 4)
    assert_close(new[4:].norm(dim=-1), torch.full((6,), 0.5))


def test_skinned_points_match_regressing_skinned_vertices():
    generator = torch.Generator().manual_seed(0)
    vertices, bones = 30, 5
    rest = torch.randn(vertices, 3, generator=generator)
    indices = torch.randint(0, bones, (vertices, 3), generator=generator)
    weights = torch.rand(vertices, 3, generator=generator)
    weights = weights / weights.sum(dim=1, keepdim=True)
    regressor = torch.rand(4, vertices, generator=generator)
    skinning = torch.eye(4).repeat(2, bones, 1, 1)
    skinning[..., :3, :] = torch.randn(2, bones, 3, 4, generator=generator)

    expected = regressor @ linear_blend_skinning(rest, indices, weights, skinning)
    found = skinned_points(
        skinning, skinned_point_weights(regressor, rest, indices, weights, bones)
    )
    torch.testing.assert_close(found, expected)


def test_turn_eyes_turns_each_eyeball_about_its_centre_and_nothing_else():
    vertices = torch.randn(ict.EYEBALLS.stop, 3, dtype=torch.float64)
    rotations = torch.tensor([[0.3, -0.2, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64)

    turned = ict.turn_eyes(vertices, rotations)

    assert_close(turned[: ict.EYEBALLS.start], vertices[: ict.EYEBALLS.start])
    left, right = ict.EYEBALL_ORDER
    assert_close(turned[right], vertices[right])
    assert_close(turned[left].mean(0), vertices[left].mean(0))
    centred = vertices[left] - vertices[left].mean(0)
    expected = centred @ roma.rotvec_to_rotmat(rotations[0]).T + vertices[left].mean(0)
    assert_close(turned[left], expected)


def test_adjacent_faces_pairs_triangles_sharing_an_edge():
    # A fan of three triangles around vertex 0: 0-1 share edge (0, 2), 1-2 share edge (0, 3).
    faces = torch.tensor([[0, 1, 2], [0, 2, 3], [0, 3, 4]])
    pairs = adjacent_faces(faces).sort(dim=1).values
    assert sorted(map(tuple, pairs.tolist())) == [(0, 1), (1, 2)]


def test_read_stl_reads_binary_and_ascii_files_alike(tmp_path):
    corners = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[1, 0, 0], [1, 1, 0], [0, 1, 0]]])
    record = np.dtype([("normal", "<f4", 3), ("corners", "<f4", (3, 3)), ("extra", "<u2")])
    rows = np.zeros(2, record)
    rows["corners"] = corners
    (tmp_path / "binary.stl").write_bytes(bytes(80) + np.uint32(2).tobytes() + rows.tobytes())
    facets = "".join(
        "facet normal 0 0 1\nouter loop\n"
        + "".join(f"vertex {x} {y} {z}\n" for x, y, z in triangle)
        + "endloop\nendfacet\n"
        for triangle in corners
    )
    (tmp_path / "ascii.stl").write_text(f"solid test\n{facets}endsolid test\n")
    for name in ("binary.stl", "ascii.stl"):
        vertices, faces = read_stl(tmp_path / name)
        assert len(vertices) == 4
        np.testing.assert_allclose(vertices[faces], corners)
