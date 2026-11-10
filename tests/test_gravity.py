import numpy as np
import pytest
import torch

from avatar_splats.export.avatar_file import AvatarFile
from avatar_splats.regions.gravity import (
    GRAVITY_SHAPES,
    GravityFit,
    GravityModel,
    gravity_weights,
    mirror_map,
    with_gravity,
)


def turned_about_x(angle: float) -> np.ndarray:
    """Gravity in the axes of a chest turned by ``angle`` about x (forward: her up to -y)."""
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return rotation.T @ np.array([0.0, 0, -1])


@pytest.mark.parametrize(
    ("gravity", "expected"),
    [
        (turned_about_x(0), [0, 0, 0, 0, 0]),
        (turned_about_x(np.pi / 2), [1, 0, 0, 0, 1]),  # leaning forward
        (turned_about_x(-np.pi / 2), [0, 1, 0, 0, 1]),  # lying on her back
        (turned_about_x(np.pi), [0, 0, 0, 0, 2]),  # upside down
        (np.array([1.0, 0, 0]), [0, 0, 1, 0, 1]),  # on her left side
    ],
)
def test_gravity_weights_as_the_viewer(gravity, expected):
    assert np.allclose(gravity_weights(gravity), expected, atol=1e-9)
    weights = gravity_weights(torch.tensor(gravity, dtype=torch.float32))
    assert torch.allclose(weights, torch.tensor(expected, dtype=torch.float32), atol=1e-6)


def symmetric_strip():
    xs = np.array([-0.1, -0.05, 0.0, 0.05, 0.1])
    vertices = np.array([(x, y, 0.0) for y in (0.0, 0.05) for x in xs], np.float32)
    faces = []
    for i in range(4):
        faces += [(i, i + 1, i + 5), (i + 1, i + 6, i + 5)]
    return vertices, np.array(faces)


def test_mirror_map_pairs_reflections():
    vertices, _ = symmetric_strip()
    mirror = mirror_map(vertices)
    assert np.allclose(vertices[mirror] * [-1, 1, 1], vertices)


def test_right_is_the_left_mirrored_and_the_prior_follows_tissue():
    vertices, faces = symmetric_strip()
    rest = torch.from_numpy(vertices)
    tissue = torch.linspace(0, 1, len(vertices))
    model = GravityModel(rest, torch.from_numpy(faces), torch.ones(len(vertices)), tissue, 0.02, 1)
    with torch.no_grad():
        model.values.normal_()
    shapes = model.shapes().detach()
    left, right = GRAVITY_SHAPES.index("breastsLeft"), GRAVITY_SHAPES.index("breastsRight")
    residuals = model.residuals().detach()
    assert torch.allclose(residuals[right], model.mirrored(residuals[left]))
    forward = GRAVITY_SHAPES.index("breastsForward")
    assert torch.allclose(residuals[forward], model.mirrored(residuals[forward]), atol=1e-6)
    prior = model.prior().detach()
    assert prior[forward, 0].abs().max() == 0 and prior[forward, -1, 1] == pytest.approx(-0.02)
    assert shapes.shape == (len(GRAVITY_SHAPES), len(vertices), 3)


def test_with_gravity_adds_body_shapes_in_format_5():
    file = AvatarFile({"format_version": 4, "head_start": 3}, {"rest_vertices": np.zeros((5, 3))})
    count = len(GRAVITY_SHAPES)
    empty = np.zeros(0)
    gravity = GravityFit(
        names=np.array(GRAVITY_SHAPES), shapes=np.ones((count, 3, 3)), sag=0.02, keys=empty,
        weights=empty, used=empty, normal_before=empty, normal_after=empty, iou_before=empty,
        iou_after=empty, rotations=empty, translations=empty, turns=empty,
    )  # fmt: skip
    out = with_gravity(file, gravity)
    assert out.metadata["format_version"] == 5 and file.metadata["format_version"] == 4
    assert out.metadata["body_shapes"]["names"] == list(GRAVITY_SHAPES)
    assert out.metadata["body_shapes"]["driver"] == {
        "kind": "gravity", "bone": "spine01", "down": [0.0, 0.0, -1.0]
    }  # fmt: skip
    assert out.arrays["body_shape_basis"].shape == (count, 3, 3)
