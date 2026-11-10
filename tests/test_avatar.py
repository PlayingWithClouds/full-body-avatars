import dataclasses

import numpy as np
import pytest
import roma
import torch
from numpy.testing import assert_array_equal
from torch.testing import assert_close

from avatar_splats.body_models.hybrid import EYES, Avatar, Rig
from avatar_splats.export import (
    AvatarRuntime,
    FaceHair,
    avatar_arrays,
    read_avatar,
    write_avatar,
)
from avatar_splats.gaussians import BoundGaussians
from avatar_splats.training import TextureRecord

EYES_ENTRY = {"iris_colour": [0.4, 0.5, 0.45], "sclera_colour": [0.9, 0.88, 0.86]}
HAIR = FaceHair(
    faces=np.array([1]),
    barycentric=np.full((1, 3), 1 / 3),
    points=np.linspace([0, 0, 0], [0.01, 0, 0.002], 6)[None],
    widths=np.array([1e-4]),
    groups=np.array([0]),
)
HAIR_ENTRY = {"points_per_strand": 6, "groups": [{"name": "lashes", "colour": [0.1, 0.1, 0.1]}]}


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
        uvs=np.array([[[0, 0], [1, 0], [0, 1]]] * 2, np.float32),
        face_textures=np.array([0, 1], np.int8),
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
def textures() -> TextureRecord:
    return TextureRecord(
        body=np.full((4, 4, 3), 0.25, np.float32),
        head=np.full((4, 20, 3), 0.75, np.float32),
        body_coverage=np.ones((4, 4), np.float32),
        head_coverage=np.ones((4, 20), np.float32),
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


def test_avatar_file_round_trip(tmp_path, avatar, gaussians, textures):
    written = avatar_arrays(avatar, gaussians.record(), textures, 2.0, EYES_ENTRY, HAIR, HAIR_ENTRY)
    write_avatar(tmp_path / "test.avatar", written)
    read = read_avatar(tmp_path / "test.avatar")
    assert read.metadata["bones"] == ["root", "head"]
    assert read.metadata["sh_degree"] == 1
    assert read.arrays["head_texture"].dtype == np.uint8
    for name, array in written.arrays.items():
        assert read.arrays[name].dtype == array.dtype
        assert_array_equal(read.arrays[name], array)


def test_eyes_are_stored_and_lose_their_gaussians(avatar, gaussians, textures):
    avatar = dataclasses.replace(avatar, face_parts=np.array([0, EYES], np.int8))
    written = avatar_arrays(avatar, gaussians.record(), textures, 2.0, EYES_ENTRY, HAIR, HAIR_ENTRY)
    assert written.metadata["eyes"] == EYES_ENTRY
    assert written.metadata["parts"][EYES] == "eyes"
    assert_array_equal(written.arrays["face_parts"], [0, EYES])
    kept = written.arrays["gaussian_faces"]
    assert 0 < len(kept) < len(gaussians.record().faces)
    assert (kept == 0).all()


def test_runtime_reproduces_the_trained_avatar(tmp_path, avatar, gaussians, textures):
    write_avatar(
        tmp_path / "test.avatar",
        avatar_arrays(avatar, gaussians.record(), textures, 2.0, EYES_ENTRY, HAIR, HAIR_ENTRY),
    )
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


def test_gaussians_are_laid_flat_on_their_triangles(gaussians):
    with torch.no_grad():
        gaussians.rotations[:] = torch.tensor([0.5, 0.5, 0.5, 0.5])  # tipped and turned
        gaussians.offsets[:] = torch.tensor([0.2, 0.0, 0.5])
        gaussians.log_scales[0, 0] = -9.0  # narrower than the disc is thick
    gaussians.lay_flat(torch.full((50,), 0.1), torch.full((50,), -5.0))
    assert_close(
        gaussians.rotations, torch.tensor([0.7071, 0, 0, 0.7071]).expand(50, 4), atol=1e-4, rtol=0
    )
    assert_close(gaussians.log_scales[:, 2], torch.full((50,), -5.0))
    assert gaussians.log_scales[0, 0] == -5.0
    assert_close(gaussians.offsets, torch.tensor([0.2, 0.0, 0.1]).expand(50, 3))
