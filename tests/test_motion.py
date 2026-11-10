import dataclasses

import numpy as np
import roma
import torch
from numpy.testing import assert_allclose, assert_array_equal
from torch.testing import assert_close

from avatar_splats.cameras import Camera
from avatar_splats.export.avatar_file import read_avatar
from avatar_splats.export.runtime import forward_kinematics
from avatar_splats.fitting.body import badly_fitted
from avatar_splats.motion.clip import (
    MOTION_MAGIC,
    MotionClip,
    bind_relative_pose,
    floor_transform,
    write_motion,
)
from avatar_splats.motion.fit import (
    _blend_rotations,
    _blend_weights,
    _windows,
    expression_mapping,
    face_expressions,
)
from avatar_splats.motion.observe import Observations
from avatar_splats.motion.placement import _cheapest_path, orientations, weak_perspective
from avatar_splats.motion.priors import JointLimits, knee_bend, overlap, self_overlap


def _bind_poses(count: int, generator: torch.Generator) -> torch.Tensor:
    poses = torch.eye(4, dtype=torch.float64).repeat(count, 1, 1)
    poses[:, :3, :3] = roma.random_rotmat(count, generator=generator, dtype=torch.float64)
    poses[:, :3, 3] = torch.randn(count, 3, generator=generator, dtype=torch.float64)
    return poses


def test_bind_relative_pose_inverts_forward_kinematics():
    generator = torch.Generator().manual_seed(0)
    parents = [-1, 0, 1, 1, 3]
    bind = _bind_poses(len(parents), generator)
    rotations = roma.random_rotmat((2, len(parents)), generator=generator, dtype=torch.float64)
    offset = torch.randn(2, 3, generator=generator, dtype=torch.float64)

    world = forward_kinematics(bind, parents, rotations, offset)
    found, found_offset = bind_relative_pose(world, bind, parents)

    assert_close(found, rotations)
    assert_close(found_offset, offset)


def test_bind_poses_are_the_rest_pose():
    generator = torch.Generator().manual_seed(1)
    parents = [-1, 0, 0]
    bind = _bind_poses(len(parents), generator)
    rotations, offset = bind_relative_pose(bind, bind, parents)
    assert_close(rotations, torch.eye(3, dtype=torch.float64).expand(3, 3, 3))
    assert_close(offset, torch.zeros(3, dtype=torch.float64))


def test_expression_mapping_follows_ict_names():
    blendshapes = ["eyeBlinkLeft", "browInnerUp", "jawLeft", "mouthSmileRight", "tongueOut"]
    expressions = ["browInnerUp_L", "browInnerUp_R", "eyeBlink_L", "jawLeft", "mouthSmile_R"]
    mapping = expression_mapping(blendshapes, expressions)
    assert_allclose(
        mapping,
        [
            [0, 0, 1, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
        ],
    )


def test_face_expressions_fill_frames_without_a_face():
    weights = np.array([[[0.2], [np.nan], [0.6]], [[np.nan], [np.nan], [np.nan]]])
    out = face_expressions(weights, ["jawOpen"], ["jawOpen"], smoothing=1)
    assert_allclose(out[0, :, 0], [0.2, 0.4, 0.6], rtol=1e-6)
    assert_allclose(out[1], 0)


def test_weak_perspective_finds_the_translation():
    camera = Camera.fixed(640, 480, 40)
    K = camera.K.numpy().astype(np.float64)
    rng = np.random.default_rng(0)
    points = rng.normal(scale=0.1, size=(17, 3))
    points -= points.mean(axis=0)
    truth = np.array([0.3, -0.2, 6.0])
    seen = points + truth
    pixels = seen[:, :2] / seen[:, 2:] * K[[0, 1], [0, 1]] + K[:2, 2]

    translation, error = weak_perspective(points, pixels, np.ones(17), K)

    assert_allclose(translation, truth, atol=0.05)
    assert error < 1.0


def test_cheapest_path_does_not_flip_for_one_doubtful_frame():
    cost = np.array([[0.0, 1.0], [0.0, 1.0], [0.6, 0.5], [0.0, 1.0]])
    transition = np.array([[0.0, 0.5], [0.5, 0.0]])
    assert_allclose(_cheapest_path(cost, transition), [0, 0, 0, 0])


def test_orientations_include_standing_facing_the_camera():
    found = orientations()
    assert any(np.allclose(r, np.eye(3)) for r in found)
    assert_allclose(np.linalg.det(found), 1)


def test_floor_transform_levels_a_tilted_floor():
    rng = np.random.default_rng(0)
    xy = rng.uniform(-2, 2, size=(400, 2))
    tilt = np.array([0.1, -0.05])
    z = xy @ tilt + 1.5
    z[::4] += rng.uniform(0.2, 0.5, size=100)  # lifted feet and jumps
    points = np.column_stack((xy, z))

    transform = floor_transform(points)

    levelled = points @ transform[:3, :3].T + transform[:3, 3]
    assert_allclose(np.quantile(levelled[:, 2], 0.5), 0, atol=0.01)
    assert np.ptp(levelled[1::4, 2]) < 0.02


def test_knees_bending_forward_cost():
    lateral = torch.tensor([1.0, 0, 0])  # the body's left, facing -y
    hip, knee = torch.tensor([0.0, 0, 1]), torch.tensor([0.0, 0, 0.5])
    backwards = torch.tensor([0.0, 0.3, 0.1])
    forwards = torch.tensor([0.0, -0.3, 0.1])
    assert knee_bend(hip, knee, backwards, lateral) == 0
    assert knee_bend(hip, knee, forwards, lateral) > 0


def test_overlap_only_between_people():
    sphere = torch.tensor([0.0, 0, 0, 0.1])
    apart = torch.stack((sphere, sphere + torch.tensor([1.0, 0, 0, 0])))[:, None, None]
    touching = torch.stack((sphere, sphere + torch.tensor([0.1, 0, 0, 0])))[:, None, None]
    same_person = torch.stack((sphere, sphere))[None, None]
    assert overlap(apart, 0.0) == 0
    assert overlap(touching, 0.0) > 0
    assert overlap(touching, 0.6) == 0
    assert overlap(same_person, 0.0) == 0


def test_windows_cover_every_frame_with_overlap():
    windows = _windows(100, 40, 10)
    assert windows[0].start == 0 and windows[-1].stop == 100
    for before, after in zip(windows, windows[1:], strict=False):
        assert after.start <= before.stop - 10
    assert _windows(30, 40, 10) == [slice(0, 30)]


def test_blend_weights_rise_across_the_overlap():
    assert_allclose(_blend_weights(slice(0, 4), done=0), [1, 1, 1, 1])
    assert_allclose(_blend_weights(slice(10, 16), done=13), [0.25, 0.5, 0.75, 1, 1, 1])


def test_blend_rotations_goes_the_short_way():
    old = np.zeros((1, 3, 1, 3), np.float32)
    new = np.zeros((1, 3, 1, 3), np.float32)
    new[..., 2] = 0.8
    mixed = _blend_rotations(old, new, np.array([0.0, 0.5, 1.0], np.float32))
    assert_allclose(mixed[0, :, 0, 2], [0, 0.4, 0.8], atol=1e-6)


def test_weak_perspective_batches_over_candidates():
    camera = Camera.fixed(640, 480, 40)
    K = camera.K.numpy().astype(np.float64)
    rng = np.random.default_rng(1)
    points = rng.normal(scale=0.2, size=(3, 17, 3))
    truth = np.array([[0.1, 0.0, 4.0], [-0.3, 0.2, 7.0], [0.0, -0.1, 5.0]])
    seen = points + truth[:, None]
    pixels = seen[..., :2] / seen[..., 2:] * K[[0, 1], [0, 1]] + K[:2, 2]

    translation, error = weak_perspective(points, pixels, np.ones((3, 17)), K)

    # Weak perspective treats each set as flat: a few per cent off in depth.
    assert_allclose(translation, truth, rtol=0.05, atol=0.02)
    assert (error < 2.0).all()


def test_mask_frames_unpack_what_was_packed():
    rng = np.random.default_rng(2)
    masks = rng.random((2, 3, 5, 13)) > 0.5
    observations = Observations(
        frame_indices=np.arange(3),
        fps=10.0,
        image_size=np.array([26, 10]),
        mask_size=np.array([13, 5]),
        masks=np.packbits(masks, axis=-1),
        boxes=np.zeros((2, 3, 4)),
        keypoints=np.zeros((2, 3, 17, 3)),
        hands=np.zeros((2, 3, 2, 21, 3)),
        blendshapes=np.zeros((2, 3, 0)),
        blendshape_names=np.array([], str),
    )
    assert (observations.mask_frames(slice(1, 3)) == masks[:, 1:3]).all()
    assert not observations.ignored_frames([0, 2]).any()
    ignored = dataclasses.replace(observations, ignored=np.packbits(~masks, axis=-1))
    assert (ignored.ignored_frames([0, 2]) == ~masks[:, [0, 2]]).all()


def test_motion_file_round_trip(tmp_path):
    rng = np.random.default_rng(3)
    clip = MotionClip(
        fps=30.0,
        bone_names=np.array(["root", "head"]),
        expression_names=np.array(["jawOpen"]),
        rotations=rng.normal(size=(2, 4, 2, 4)).astype(np.float32),
        root_translation=rng.normal(size=(2, 4, 3)).astype(np.float32),
        expressions=rng.random((2, 4, 1)).astype(np.float32),
        seen=np.ones((2, 4), bool),
        camera_to_world=np.eye(4),
        fov_y_degrees=40.0,
    )
    write_motion(tmp_path / "clip.motion", clip)
    file = read_avatar(tmp_path / "clip.motion", MOTION_MAGIC)
    assert file.metadata["people"] == 2 and file.metadata["frames"] == 4
    assert file.metadata["bones"] == ["root", "head"]
    assert_allclose(file.arrays["rotations"], clip.rotations)
    assert_allclose(file.arrays["root_translation"], clip.root_translation)


def test_joint_limits_in_the_bones_own_axes():
    class Body:
        bone_labels = ("root", "lowerleg01.L", "lowerleg01.R")
        num_bones = 3

    # the left knee's own axes: the world's turned a quarter about z, its x along world y
    axes = torch.eye(4).repeat(3, 1, 1)
    axes[1, :3, :3] = roma.rotvec_to_rotmat(torch.tensor([0, 0, np.pi / 2]))
    limits = JointLimits(Body(), axes)
    rotations = torch.zeros(5, 3, 3)
    rotations[0, 0] = torch.tensor([3.0, 0, 0])  # the root is free
    rotations[1, 1, 1] = np.radians(90)  # the left knee bends 90 degrees about its own x
    rotations[2, 1, 1] = -np.radians(24)  # and 24 the other way: 10 past its 14
    rotations[3, 2, 2] = np.radians(8)  # the right knee sideways: 3 past its 5
    rotations[4, 2, 1] = np.radians(10)  # its twist, mirrored: 5 past
    assert_allclose(
        limits(rotations).numpy(), np.radians([0, 0, 10, 3, 5]) ** 2, rtol=1e-4, atol=1e-8
    )


def test_self_overlap_only_between_far_bones():
    spheres = torch.tensor([[0.0, 0, 0, 0.1], [0.1, 0, 0, 0.1], [0.0, 0.1, 0, 0.1]])
    far = torch.tensor([[False, True, False], [True, False, False], [False, False, False]])
    # spheres 0 and 1: 0.1 apart, reach 0.7 * 0.2 = 0.14, so 0.04 deep; 0 and 2 are near bones
    assert_allclose(self_overlap(spheres, far, 0.3).item(), 0.04**2, rtol=1e-5)
    assert self_overlap(spheres, torch.zeros_like(far), 0.3).item() == 0


def test_badly_fitted_takes_outliers_up_to_a_share():
    iou = np.array([0.85, 0.84, 0.86, 0.83, 0.85, 0.40, 0.84, 0.86, 0.30, np.nan, 0.85, 0.84])
    assert_array_equal(np.flatnonzero(badly_fitted(iou, 2.0, 1 / 3)), [5, 8, 9])
    assert_array_equal(np.flatnonzero(badly_fitted(iou, 2.0, 1 / 6)), [8, 9])  # the worst two
