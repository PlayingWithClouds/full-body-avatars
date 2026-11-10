import math

import numpy as np
import torch

from avatar_splats.cameras import Camera
from avatar_splats.fitting.losses import depth_loss, keypoint_depth_loss, normal_loss
from avatar_splats.preprocess.surface import _place, lift_keypoints, pointmap_focal
from avatar_splats.rendering.surface import clip_space


def test_pointmap_focal_recovers_the_focal_length():
    generator = torch.Generator().manual_seed(0)
    height, width, focal, cx, cy = 60, 40, 55.0, 22.0, 27.0
    v, u = torch.meshgrid(torch.arange(height).float(), torch.arange(width).float(), indexing="ij")
    z = 2 + torch.rand(height, width, generator=generator)
    points = torch.stack(((u - cx) * z / focal, (v - cy) * z / focal, z))
    points[:, :5, :5] = torch.tensor([3.0, -2.0, 0.5])[:, None, None]  # wild guesses
    assert math.isclose(pointmap_focal(points), focal, rel_tol=1e-3)
    selected = torch.zeros(height, width, dtype=torch.bool)
    selected[10:50, 5:35] = True
    assert math.isclose(pointmap_focal(points, selected), focal, rel_tol=1e-3)
    assert math.isnan(pointmap_focal(points, torch.zeros_like(selected)))


def test_place_crops_to_the_map():
    values = torch.arange(2 * 8 * 6, dtype=torch.float32).reshape(2, 8, 6)
    # a crop half outside on the left, at the maps' half size
    placed, window = _place(values, (-6, 0, 6, 16), 0.5, 10, 10)
    assert window == (slice(0, 8), slice(0, 3))
    assert placed.shape == (2, 8, 3)
    assert torch.equal(placed, values[:, :, 3:])
    assert _place(values, (30, 30, 42, 46), 0.5, 10, 10) is None


def test_normal_loss():
    normals = torch.nn.functional.normalize(torch.randn(2, 4, 5, 3), dim=-1)
    region = torch.ones(2, 4, 5)
    assert normal_loss(normals, normals, region).item() < 1e-6
    assert math.isclose(normal_loss(normals, -normals, region).item(), 2, rel_tol=1e-5)


def test_depth_loss_compares_relief_only():
    rendered = (1 + torch.rand(2, 6, 7)).requires_grad_()
    region = torch.ones(2, 6, 7)
    targets = rendered.detach() * torch.tensor([2.0, 0.5])[:, None, None]
    targets[0, 0, 0] = torch.nan  # unknown: left out, without poisoning the gradient
    loss = depth_loss(rendered, targets, region)
    assert loss.item() < 1e-6
    flattened = rendered.detach().mean() * torch.ones_like(targets)
    loss = depth_loss(rendered, flattened, region)
    loss.backward()
    assert loss.item() > 0.01
    assert torch.isfinite(rendered.grad).all()


def test_keypoint_depth_loss_is_lopsided():
    generator = torch.Generator().manual_seed(0)
    depth = (2 + torch.rand(3, 17, generator=generator)).requires_grad_()
    insets = torch.full((17,), 0.05)
    targets = 0.7 * (depth.detach() - insets)  # the skin's depth at another scale
    targets[0, :5] = torch.nan
    targets[2, 1:] = torch.nan  # one keypoint alone says nothing about order
    assert keypoint_depth_loss(depth, targets, insets).item() < 1e-5
    behind, nearer = targets.clone(), targets.clone()
    behind[1, 13] -= 0.5 / 0.7  # the knee's pixel shows something 0.5 m in front of it
    nearer[1, 13] += 0.5 / 0.7  # the knee is 0.5 m nearer than the skin at its pixel
    behind_loss = keypoint_depth_loss(depth, behind, insets)
    nearer_loss = keypoint_depth_loss(depth, nearer, insets)
    assert 0.001 < behind_loss.item() < nearer_loss.item() / 3
    nearer_loss.backward()
    assert torch.isfinite(depth.grad).all()
    assert (depth.grad[2] == 0).all()
    face = torch.ones(17)
    face[:5] = 0.2
    assert keypoint_depth_loss(depth, nearer, insets, face) > nearer_loss


def test_lift_keypoints_on_the_person_only():
    # frames 40 x 20, the map at half the size
    keypoints = np.full((1, 17, 3), np.nan)
    keypoints[0, 13] = (5, 10, 0.9)  # on the person
    keypoints[0, 14] = (15, 10, 0.9)  # off the person
    keypoints[0, 15] = (5, 30, 0.1)  # unsure
    depth = np.full((1, 20, 10), np.nan, np.float16)
    depth[0, :, :5] = 1.5
    lifted = lift_keypoints(keypoints, depth, (20, 40), min_confidence=0.3, radius=1)
    assert lifted[0, 13] == 1.5
    assert np.isnan(lifted[0, 14:]).all()
    assert np.isnan(lifted[0, :13]).all()


def test_clip_space_matches_the_projection():
    camera = Camera.fixed(64, 48, 40.0)
    points = torch.tensor([[[0.1, 2.0, 0.2], [-0.3, 3.0, -0.1]]])
    clip = clip_space(points, camera, 0.1, 20.0)
    pixels = (clip[..., :2] / clip[..., 3:] + 1) / 2 * torch.tensor([64.0, 48.0])
    assert torch.allclose(pixels, camera.project(points), atol=1e-4)
    depth = clip[..., 2] / clip[..., 3]
    assert ((depth > -1) & (depth < 1)).all()
