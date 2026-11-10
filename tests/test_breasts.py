import numpy as np
import pytest
import torch

from avatar_splats.export.avatar_file import AvatarFile
from avatar_splats.regions.breasts import (
    BREAST_CHANGES,
    BREAST_PHENOTYPES,
    BreastFit,
    averaging_matrix,
    clip,
    nipple_error,
    project,
    smooth,
    torso_faces,
    with_breasts,
)
from avatar_splats.regions.closeups import (
    crop_intrinsics,
    cut,
    intrinsics,
    nipple_centres,
    square_crop,
)


def test_crop_intrinsics_project_like_the_frame():
    K = intrinsics(40.0, 1920, 1080)
    point = np.array([0.1, -0.05, 2.0])
    frame_pixel = (K @ point)[:2] / point[2]
    crop = square_crop(np.array([900.0, 400, 1100, 560]), 2.0)
    assert crop[2] - crop[0] == pytest.approx(400) and crop[:2].tolist() == [800, 280]
    C = crop_intrinsics(K, crop, 256)
    crop_pixel = (C @ point)[:2] / point[2]
    assert np.allclose(crop_pixel, (frame_pixel - crop[:2]) * 256 / 400)


def test_cut_pads_past_the_frame():
    frame = np.full((100, 100, 3), 200, np.uint8)
    out = cut(frame, np.array([-50.0, -50, 50, 50]), 20)
    assert out.shape == (20, 20, 3) and out[0, 0].sum() == 0 and out[-1, -1, 0] == 200


def test_nipple_centres_merge_doubles_and_skip_areolae_around_nipples():
    def disc(x, y, r):
        yy, xx = np.mgrid[:200, :200]
        return (xx - x) ** 2 + (yy - y) ** 2 <= r * r

    found = nipple_centres(
        [disc(50, 50, 3), disc(52, 51, 3)], [disc(53, 52, 12), disc(150, 60, 10)]
    )
    assert np.isfinite(found).all()
    assert np.allclose(found[0], (51, 50.5), atol=1) and np.allclose(found[1], (150, 60), atol=1)
    assert np.isnan(nipple_centres([], [])).all()


def test_nipple_error_pairs_and_singles():
    model = torch.tensor([[[10.0, 0], [20.0, 0]]])
    both = torch.tensor([[[20.0, 0], [10.0, 3]]])  # crossed, one 3 px off
    assert nipple_error(model, both).item() == pytest.approx(1.5)
    one = torch.tensor([[[19.0, 0], [float("nan")] * 2]])
    assert nipple_error(model, one).item() == pytest.approx(1.0)


def test_clip_agrees_with_projection():
    K = torch.tensor([[[300.0, 0, 120], [0, 300, 140], [0, 0, 1]]])
    points = torch.tensor([[[0.1, -0.2, 2.0], [-0.3, 0.05, 1.5]]])
    pixels = project(points, K)
    c = clip(points, K, 256)
    ndc = c[..., :2] / c[..., 3:]
    assert torch.allclose((ndc + 1) / 2 * 256, pixels, atol=1e-3)


def test_smoothing_keeps_constants_and_spreads_spikes():
    faces = torch.tensor([[0, 1, 2], [1, 3, 2], [2, 3, 4]])
    matrix = averaging_matrix(faces, 5)
    ones = torch.ones(5, 1)
    assert torch.allclose(smooth(ones, matrix, 3), ones)
    spike = torch.zeros(5, 1)
    spike[0] = 1
    spread = smooth(spike, matrix, 2)
    assert spread[0] < 1 and spread[4] > 0 and spread.sum() > 0


def test_torso_faces_leave_out_limbs():
    faces = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]])
    indices = np.array([[0, 1], [0, 1], [0, 1], [1, 0], [1, 0]])
    weights = np.array([[0.9, 0.1]] * 3 + [[0.8, 0.2]] * 2)
    labels = ["spine01", "upperarm01.L"]
    assert torso_faces(faces, indices, weights, labels, head_start=5).tolist() == [0]


def test_with_breasts_moves_the_body_only():
    rest = np.zeros((6, 3), np.float32)
    file = AvatarFile({"head_start": 4}, {"rest_vertices": rest})
    change = np.full((4, 3), 0.01, np.float32)
    count = len(BREAST_PHENOTYPES) + len(BREAST_CHANGES)
    empty = np.zeros(0)
    fit = BreastFit(
        change=change, weights=np.ones(4), nipples=np.array([0, 1]),
        controls=np.linspace(0, 1, count), controls_before=np.zeros(count), keys=empty,
        rotations=empty, translations=empty, turns=empty, tilt=empty, gravity=empty,
        shaping=empty, iou_before=empty, iou_after=empty, normal_before=empty,
        normal_after=empty, nipple_before=empty, nipple_after=empty,
    )  # fmt: skip
    out = with_breasts(file, fit)
    assert np.allclose(out.arrays["rest_vertices"][:4], 0.01)
    assert np.allclose(out.arrays["rest_vertices"][4:], 0) and np.allclose(rest, 0)
    assert out.metadata["breasts"]["controls"]["cupsize"] == 0 and "breasts" not in file.metadata
