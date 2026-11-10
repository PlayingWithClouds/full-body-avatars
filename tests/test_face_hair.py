import numpy as np
import torch
from numpy.testing import assert_allclose

from avatar_splats.body_models import ict
from avatar_splats.body_models.hybrid import HEAD_SKIN, Avatar
from avatar_splats.body_models.mesh import triangle_frames
from avatar_splats.export.face_hair import _bind, lash_cards


def grid_card(rows: int = 6, columns: int = 19) -> tuple[np.ndarray, np.ndarray]:
    """A card standing up from the line y = z = 0, triangulated, its vertices shuffled."""
    x, z = np.meshgrid(np.linspace(0, 0.03, columns), np.linspace(0, 0.008, rows))
    points = np.stack((x, 0.002 * x / 0.03, z), axis=-1).reshape(-1, 3)
    index = np.arange(rows * columns).reshape(rows, columns)
    quads = np.stack(
        (index[:-1, :-1], index[:-1, 1:], index[1:, 1:], index[1:, :-1]), axis=-1
    ).reshape(-1, 4)
    triangles = np.concatenate((quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]))
    order = np.random.default_rng(0).permutation(len(points))
    return points[order], np.argsort(order)[triangles]


def test_lash_cards_come_out_as_grids_from_root_to_tip():
    card, triangles = grid_card()
    vertices = np.zeros((ict.LASH_CARDS.stop, 3))
    first = ict.LASH_CARDS.start
    vertices[first : first + len(card)] = card
    vertices[ict.LASH_CARDS.start + len(card) :] = 1.0  # the other cards, far away
    skin = np.stack((np.linspace(-0.01, 0.04, 200), np.zeros(200), np.zeros(200)), axis=1)
    skin[:, 1] = 0.002 * skin[:, 0] / 0.03

    (grid,) = [g for g in lash_cards(vertices, triangles + first, skin) if g.shape == (6, 19, 3)]
    assert_allclose(grid[:, :, 2], np.broadcast_to(np.linspace(0, 0.008, 6)[:, None], (6, 19)))
    along = grid[0, :, 0]
    assert (np.diff(along) > 0).all() or (np.diff(along) < 0).all()
    assert_allclose(grid[:, 0, 0], grid[0, 0, 0])  # each column one lash


def test_strands_are_stored_in_their_triangles_frames():
    vertices = np.array([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0.002], [0.01, 0.01, 0]], np.float64)
    faces = np.array([[0, 1, 2], [1, 3, 2]])
    avatar = Avatar(
        rest_vertices=vertices,
        faces=faces,
        face_parts=np.full(2, HEAD_SKIN, np.int8),
        uvs=np.zeros((2, 3, 2), np.float32),
        face_textures=np.ones(2, np.int8),
        bone_indices=np.zeros((4, 1), int),
        bone_weights=np.ones((4, 1), np.float32),
        bone_labels=np.array(["head"]),
        bone_parents=np.array([-1]),
        bind_poses=np.eye(4, dtype=np.float32)[None],
        head_start=0,
        expression_names=np.array([]),
        expression_basis=np.zeros((0, 4, 3), np.float32),
    )
    roots = np.array([[0.002, 0.003, 0.0006], [0.008, 0.007, 0.0004]])
    strands = roots[:, None] + np.linspace(0, 1, 6)[None, :, None] * np.array([0, 0, 0.005])
    hair = _bind(avatar, strands, np.full(2, 1e-4), np.zeros(2, np.uint8))

    frames = triangle_frames(torch.from_numpy(vertices), torch.from_numpy(faces[hair.faces]))[0]
    on_mesh = np.einsum("sc,scd->sd", hair.barycentric, vertices[faces[hair.faces]])
    rebuilt = on_mesh[:, None] + np.einsum("sij,spj->spi", frames.numpy(), hair.points)
    assert_allclose(on_mesh, roots, atol=2e-4)
    assert_allclose(rebuilt - rebuilt[:, :1], strands - strands[:, :1], atol=1e-9)
    assert_allclose(hair.points[:, 0], 0, atol=1e-12)
