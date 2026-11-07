import numpy as np
import pytest
import roma
import torch
from numpy.testing import assert_array_equal
from torch.testing import assert_close

from avatar_splats.body_models.hybrid import Avatar, Rig
from avatar_splats.export import AvatarRuntime, avatar_arrays, read_avatar, write_avatar
from avatar_splats.gaussians import BoundGaussians


@pytest.fixture
def avatar() -> Avatar:
    """Two triangles hanging from a two-bone chain, the second with one expression."""
    bind = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
    bind[1, :3, 3] = [0, 0, 1]
    return Avatar(
        rest_vertices=np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1]], np.float32
        ),
        faces=np.array([[0, 1, 2], [3, 4, 5]]),
        face_parts=np.array([0, 1], np.int8),
        bone_indices=np.array([[0, 1]] * 3 + [[1, 0]] * 3),
        bone_weights=np.array([[1, 0]] * 3 + [[1, 0]] * 3, np.float32),
        bone_labels=np.array(["root", "head"]),
        bone_parents=np.array([-1, 0]),
        bind_poses=bind,
        head_start=3,
        expression_names=np.array(["jawOpen"]),
        expression_basis=np.array([[[0, 0, 0.5], [0, 0, 0], [0, 0, 0]]], np.float32),
    )


@pytest.fixture
def gaussians(avatar: Avatar) -> BoundGaussians:
    gaussians = BoundGaussians.initialize(
        torch.from_numpy(avatar.rest_vertices), torch.from_numpy(avatar.faces), torch.ones(2), 50, 1
    )
    with torch.no_grad():
        gaussians.offsets.normal_(0, 0.1)
        gaussians.rotations.normal_()
    return gaussians


def test_rig_applies_expressions_then_skinning(avatar):
    rig = Rig(avatar)
    skinning = torch.eye(4).repeat(1, 2, 1, 1)
    skinning[0, 1, :3, 3] = torch.tensor([1.0, 0, 0])
    vertices = rig(skinning, torch.tensor([[1.0]]))[0]
    assert_close(vertices[:3], torch.from_numpy(avatar.rest_vertices[:3]))
    assert_close(vertices[3], torch.tensor([1.0, 0, 1.5]))


def test_gaussians_follow_their_triangles(avatar, gaussians):
    rig = Rig(avatar)
    rest = rig.rest_vertices
    means, rotations, scales = gaussians.place(rest, rig.faces)

    rotation = roma.rotvec_to_rotmat(torch.tensor([0.2, -0.4, 0.9]))
    shift = torch.tensor([1.0, -2, 0.5])
    moved = 2.5 * rest @ rotation.T + shift
    moved_means, moved_rotations, moved_scales = gaussians.place(moved, rig.faces)

    assert_close(moved_means, 2.5 * means @ rotation.T + shift, atol=1e-5, rtol=0)
    assert_close(moved_scales, 2.5 * scales)
    before = roma.unitquat_to_rotmat(rotations[:, [1, 2, 3, 0]])
    after = roma.unitquat_to_rotmat(moved_rotations[:, [1, 2, 3, 0]])
    assert_close(after, rotation @ before, atol=1e-5, rtol=0)


def test_avatar_file_round_trip(tmp_path, avatar, gaussians):
    written = avatar_arrays(avatar, gaussians.record())
    write_avatar(tmp_path / "test.avatar", written)
    read = read_avatar(tmp_path / "test.avatar")
    assert read.metadata["bones"] == ["root", "head"]
    assert read.metadata["sh_degree"] == 1
    for name, array in written.arrays.items():
        assert read.arrays[name].dtype == array.dtype
        assert_array_equal(read.arrays[name], array)


def test_runtime_reproduces_the_trained_avatar(tmp_path, avatar, gaussians):
    write_avatar(tmp_path / "test.avatar", avatar_arrays(avatar, gaussians.record()))
    runtime = AvatarRuntime(read_avatar(tmp_path / "test.avatar"))

    # Bones at rest pose the mesh in its rest pose.
    rest = runtime.pose(torch.eye(3).repeat(1, 2, 1, 1))[0]
    assert_close(rest, torch.from_numpy(avatar.rest_vertices))

    # A rotated root carries its child along.
    rotations = torch.eye(3).repeat(1, 2, 1, 1)
    rotations[0, 0] = roma.rotvec_to_rotmat(torch.tensor([0, 0, torch.pi / 2]))
    posed = runtime.pose(rotations)[0]
    assert_close(posed[4], torch.tensor([0.0, 1, 1]), atol=1e-6, rtol=0)

    means, *_ = runtime.place(rest)
    expected, *_ = gaussians.place(rest, torch.from_numpy(avatar.faces))
    assert_close(means, expected)


def test_flatness_tells_lying_from_standing_gaussians(gaussians):
    with torch.no_grad():
        gaussians.log_scales[:] = torch.tensor([0.0, 0.0, -3.0])  # thin along the third axis
        gaussians.rotations[:] = torch.tensor([1.0, 0, 0, 0])  # lying flat on the triangle
        gaussians.rotations[:10] = torch.tensor([0.7071, 0.7071, 0, 0])  # tipped up 90 degrees
    flatness = gaussians.flatness()
    assert_close(flatness[10:], torch.ones(40))
    assert_close(flatness[:10], torch.zeros(10), atol=1e-4, rtol=0)
