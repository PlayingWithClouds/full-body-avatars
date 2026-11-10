import numpy as np
import torch

from avatar_splats.export.avatar_file import AvatarFile
from avatar_splats.regions.texture import (
    TEXTURE_NAME,
    BreastTexture,
    harmonic_uvs,
    push_pull,
    region_faces,
    with_breast_texture,
)


def bump(n: int = 12):
    """A disc of triangles over a bump: x across, z up, y out of it (towards -y)."""
    xs, zs = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n))
    keep = xs**2 + zs**2 <= 1.0
    index = -np.ones(xs.shape, int)
    index[keep] = np.arange(keep.sum())
    vertices = np.stack(
        [xs[keep], -0.6 * np.exp(-3 * (xs[keep] ** 2 + zs[keep] ** 2)), zs[keep]], 1
    )
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = index[i, j], index[i, j + 1], index[i + 1, j], index[i + 1, j + 1]
            if min(a, b, d) >= 0:
                faces.append((a, b, d))
            if min(a, d, c) >= 0:
                faces.append((a, d, c))
    return vertices, np.array(faces)


def signed_areas(corners: np.ndarray) -> np.ndarray:
    a, b = corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
    return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]


def test_harmonic_uvs_flatten_without_flips_and_follow_x_and_z():
    vertices, faces = bump()
    uvs = harmonic_uvs(vertices, faces)
    assert uvs.min() >= 0 and uvs.max() <= 1
    areas = signed_areas(uvs)
    assert (np.sign(areas) == np.sign(areas[0])).all() and (np.abs(areas) > 0).all()
    corners = vertices[faces]
    assert np.corrcoef(uvs[..., 0].ravel(), corners[..., 0].ravel())[0, 1] > 0.95
    assert np.corrcoef(uvs[..., 1].ravel(), corners[..., 2].ravel())[0, 1] > 0.95


def test_region_faces_are_body_triangles_near_the_breasts():
    faces = np.array([[0, 1, 2], [1, 2, 3], [3, 4, 5]])
    weights = np.array([0.0, 0.0, 0.0, 0.5])
    assert region_faces(faces, weights, head_start=4).tolist() == [1]


def test_push_pull_fills_holes_and_keeps_known_texels():
    values = torch.zeros(8, 8, 1)
    weights = torch.zeros(8, 8)
    values[2, 2], weights[2, 2] = 3.0, 1.0
    values[6, 6], weights[6, 6] = 1.0, 1.0
    filled = push_pull(values, weights)
    assert filled[2, 2, 0] == 3 and filled[6, 6, 0] == 1
    assert 1 <= filled[0, 0, 0] <= 3 and 1 <= filled[7, 7, 0] <= 3


def test_with_breast_texture_points_the_region_at_the_new_texture():
    file = AvatarFile(
        {"textures": ["body_texture", "head_texture"]},
        {"face_textures": np.zeros(3, np.uint8), "uvs": np.zeros((3, 3, 2), np.float32)},
    )
    image = np.zeros((4, 4, 3))
    image[0] = 1  # row 0 is v = 0: the file's last row
    texture = BreastTexture(
        faces=np.array([1]), uvs=np.full((1, 3, 2), 0.5, np.float32), texture=image, base=image,
        coverage=np.zeros((4, 4)), inside=np.ones((4, 4), bool), keys=np.zeros(0),
    )  # fmt: skip
    out = with_breast_texture(file, texture)
    assert out.metadata["textures"] == ["body_texture", "head_texture", TEXTURE_NAME]
    assert out.arrays["face_textures"].tolist() == [0, 2, 0]
    assert np.allclose(out.arrays["uvs"][1], 0.5) and np.allclose(file.arrays["uvs"], 0)
    assert (out.arrays[TEXTURE_NAME][-1] == 255).all() and (out.arrays[TEXTURE_NAME][0] == 0).all()
