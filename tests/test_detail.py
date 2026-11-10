import numpy as np
import pytest

from avatar_splats.body_models.mesh import boundary_loops
from avatar_splats.export.avatar_file import AvatarFile
from avatar_splats.regions.detail import extended, refine_region


def grid_file(n: int = 4, head: int = 2):
    """A flat n by n grid of body vertices, then ``head`` head vertices with one head triangle."""
    xs, ys = np.meshgrid(np.arange(n, dtype=np.float32), np.arange(n, dtype=np.float32))
    body = np.stack([xs.ravel(), ys.ravel(), np.zeros(n * n, np.float32)], 1)
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = i * n + j, i * n + j + 1, (i + 1) * n + j, (i + 1) * n + j + 1
            faces += [(a, b, d), (a, d, c)]
    count = n * n
    rest = np.concatenate(
        [body, np.array([[0, 0, 5], [1, 0, 5], [0, 1, 5]], np.float32)[: head + 1]]
    )
    faces.append((count, count + 1, min(count + 2, len(rest) - 1)))
    faces = np.array(faces, np.uint32)
    uvs = rest[faces][..., :2] / n
    vertices = len(rest)
    arrays = {
        "rest_vertices": rest, "faces": faces, "uvs": uvs.astype(np.float32),
        "face_textures": np.zeros(len(faces), np.uint8),
        "face_parts": np.arange(len(faces)).astype(np.uint8),
        "bone_indices": np.tile(np.array([[0, 1]], np.uint16), (vertices, 1)),
        "bone_weights": np.tile(np.array([[0.75, 0.25]], np.float32), (vertices, 1)),
        "body_shape_basis": np.ones((2, count, 3), np.float32),
        "expression_basis": np.zeros((1, vertices - count, 3), np.float32),
    }  # fmt: skip
    metadata = {"head_start": count, "eyes": {"eyeballs": [{"apex": count, "top": count + 1}]}}
    return AvatarFile(metadata, arrays)


def test_refinement_splits_red_green_and_keeps_the_mesh_closed():
    file = grid_file()
    out, refinement = refine_region(file, np.array([8]))  # one triangle in the middle
    faces = out.arrays["faces"].astype(np.int64)
    before = len(boundary_loops(file.arrays["faces"][:-1].astype(np.int64))[0])
    head_start = out.metadata["head_start"]
    body = faces[(faces < head_start).all(axis=1)]
    assert len(boundary_loops(body)) == 1  # no cracks: the green triangles close the red ones
    assert len(boundary_loops(body)[0]) == before
    assert head_start == file.metadata["head_start"] + len(refinement.parents) == 16 + 3
    assert len(faces) == len(file.arrays["faces"]) + 3 + 3  # 4 children, 3 neighbours in 2
    # the head kept its triangle index, its vertices moved past the new ones
    assert faces[len(file.arrays["faces"]) - 1].tolist() == [19, 20, 21]
    assert out.metadata["eyes"]["eyeballs"][0] == {"apex": 19, "top": 20}
    assert np.allclose(out.arrays["rest_vertices"][19:], file.arrays["rest_vertices"][16:])


def test_refinement_mixes_every_per_vertex_array():
    file = grid_file()
    out, refinement = refine_region(file, np.array([8]))
    new = slice(16, 19)
    flat = out.arrays["rest_vertices"][new]
    assert np.allclose(flat[:, 2], 0)  # a flat grid stays flat
    assert np.allclose(out.arrays["bone_weights"][new].sum(1), 1)
    assert out.arrays["body_shape_basis"].shape == (2, 19, 3)
    assert np.allclose(out.arrays["body_shape_basis"], 1)
    assert out.arrays["expression_basis"].shape == file.arrays["expression_basis"].shape
    weights = np.arange(16, dtype=np.float32)
    assert extended(weights, refinement).shape == (19,)
    uvs = out.arrays["uvs"]
    positions = out.arrays["rest_vertices"][out.arrays["faces"].astype(np.int64)][..., :2] / 4
    body = (out.arrays["faces"] < 19).all(axis=1)
    assert np.allclose(uvs[body], positions[body], atol=0.08)  # Loop's rule moves a little


def test_refinement_refuses_the_head():
    file = grid_file()
    with pytest.raises(ValueError):
        refine_region(file, np.array([len(file.arrays["faces"]) - 1]))


def test_with_relief_moves_the_body_only():
    from avatar_splats.regions.detail import Relief, with_relief

    file = grid_file()
    change = np.full((16, 3), 0.002, np.float32)
    empty = np.zeros(0)
    out = with_relief(
        file, Relief(change=change, keys=empty, normal_before=empty, normal_after=empty)
    )
    moved = out.arrays["rest_vertices"][:16] - file.arrays["rest_vertices"][:16]
    assert np.allclose(moved, 0.002, atol=1e-6)
    assert np.allclose(out.arrays["rest_vertices"][16:], file.arrays["rest_vertices"][16:])


def test_refine_breasts_refines_twice_near_the_nipples():
    from avatar_splats.regions.detail import refine_breasts

    file = grid_file(6)
    weights = np.zeros(36, np.float32)
    weights[[14, 15, 20, 21]] = 1.0
    out, extended_weights = refine_breasts(file, weights, np.array([14, 21]), reach=1.5)
    head_start = out.metadata["head_start"]
    assert head_start > 36 + 10 and len(extended_weights) == head_start
