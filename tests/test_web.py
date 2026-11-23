from collections import Counter

import numpy as np

from avatar_splats.export.avatar_file import AvatarFile
from avatar_splats.export.web import smooth_crotch, web_avatar

PARTS = ["body", "head_skin", "mouth", "eyes", "hands", "genitals", "canals"]


def _grid(n: int) -> tuple[np.ndarray, np.ndarray]:
    """An ``n`` by ``n`` grid of vertices on the unit square, z = 0, and its triangles."""
    x, y = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n), indexing="ij")
    vertices = np.stack([x.ravel(), y.ravel(), np.zeros(n * n)], axis=1)
    index = np.arange(n * n).reshape(n, n)
    a, b, c, d = index[:-1, :-1], index[1:, :-1], index[1:, 1:], index[:-1, 1:]
    faces = np.concatenate(
        [np.stack([a, b, c], -1).reshape(-1, 3), np.stack([a, c, d], -1).reshape(-1, 3)]
    )
    return vertices, faces


def _avatar() -> dict[str, np.ndarray]:
    """A flat sheet of body with a bumped patch of genitals in its middle, a hole in the patch's
    centre, and a canal: the hole's rim pushed down into a tube."""
    vertices, faces = _grid(13)
    radius = np.linalg.norm(vertices[:, :2], axis=1)
    centres = vertices[faces].mean(axis=1)
    centre_radius = np.linalg.norm(centres[:, :2], axis=1)
    part = np.where(centre_radius < 0.55, PARTS.index("genitals"), PARTS.index("body"))
    part[centre_radius < 0.2] = PARTS.index("canals")
    vertices[radius < 0.4, 2] = 0.3 * (0.4 - radius[radius < 0.4])
    vertices[radius < 0.15, 2] = -0.5
    count = len(faces)
    return {
        "rest_vertices": vertices.astype(np.float32),
        "faces": faces.astype(np.uint32),
        "uvs": np.zeros((count, 3, 2), np.float32),
        "face_textures": np.zeros(count, np.uint8),
        "face_parts": part.astype(np.uint8),
        "mouth_regions": np.zeros(count, np.uint8),
        "bone_indices": np.zeros((len(vertices), 8), np.uint16),
        "bone_weights": np.tile(np.eye(8, dtype=np.float32)[0], (len(vertices), 1)),
        "expression_basis": np.ones((2, len(vertices), 3), np.float32),
        "hair_faces": np.array([0, count - 1], np.uint32),
    }


def _boundary_edges(faces: np.ndarray) -> int:
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), 1)
    return sum(1 for n in Counter(map(tuple, edges)).values() if n == 1)


def test_smooth_crotch_drops_canals_and_closes_their_openings():
    arrays = _avatar()
    out = smooth_crotch(arrays, PARTS)
    assert not np.any(out["face_parts"] == PARTS.index("canals"))
    # The sheet's own edge is the only boundary left: the opening is closed.
    assert _boundary_edges(out["faces"]) == _boundary_edges(_grid(13)[1])
    for name in ("uvs", "face_textures", "face_parts", "mouth_regions"):
        assert len(out[name]) == len(out["faces"])
    added = len(out["rest_vertices"]) - len(arrays["rest_vertices"])
    assert added == 1
    assert out["expression_basis"].shape[1] == len(out["rest_vertices"])
    assert np.all(out["expression_basis"][:, -added:] == 0)


def test_smooth_crotch_flattens_the_patch_between_its_pinned_edge():
    out = smooth_crotch(_avatar(), PARTS)
    # Pinned to a flat sheet, the least-bending surface is the plane.
    used = np.unique(out["faces"])
    assert np.abs(out["rest_vertices"][used, 2]).max() < 1e-4


def test_smooth_crotch_keeps_the_winding_of_the_surface():
    out = smooth_crotch(_avatar(), PARTS)
    corners = out["rest_vertices"][out["faces"].astype(np.int64)]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    assert np.all(np.sign(normals[:, 2]) == np.sign(normals[0, 2]))


def test_smooth_crotch_renumbers_face_hair():
    arrays = _avatar()
    out = smooth_crotch(arrays, PARTS)
    kept = arrays["face_parts"] != PARTS.index("canals")
    last = int(np.flatnonzero(kept)[-1])
    assert out["hair_faces"][1] == np.cumsum(kept)[last] - 1


def test_web_avatar_keeps_named_expressions_and_moves_textures_out():
    arrays = _avatar()
    arrays["body_texture"] = np.zeros((4, 4, 3), np.uint8)
    metadata = {"expressions": ["a", "b"], "parts": PARTS, "textures": ["body_texture"]}
    file, images = web_avatar(AvatarFile(metadata, arrays), ["b"])
    assert file.metadata["expressions"] == ["b"]
    assert file.arrays["expression_basis"].shape[0] == 1
    assert "body_texture" not in file.arrays
    assert file.metadata["web_textures"] == {"body_texture": "body_texture.webp"}
    assert images["body_texture"].shape == (4, 4, 3)


def test_smooth_crotch_leaves_a_body_without_genitals_alone():
    arrays = _avatar()
    arrays["face_parts"] = np.zeros_like(arrays["face_parts"])
    assert smooth_crotch(arrays, PARTS) is arrays
