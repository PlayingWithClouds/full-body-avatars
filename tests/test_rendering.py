import torch
from torch.testing import assert_close

from avatar_splats.cameras import Camera
from avatar_splats.rendering import (
    behind_mesh,
    facing_away,
    rasterize,
    render_mesh,
    sample_texture,
)

SQUARE = torch.tensor([[0, 1, 2], [0, 2, 3]])


def square(depth: float, size: float) -> torch.Tensor:
    """A square from (0, 0) to (size, size) pixels at a constant depth."""
    corners = torch.tensor([[0.0, 0], [size, 0], [size, size], [0, size]])
    return torch.cat((corners, torch.full((4, 1), depth)), dim=1)


def test_a_square_covers_exactly_its_pixels():
    fragments = rasterize(square(1, 6), SQUARE, 10, 8)
    mask = torch.zeros(8, 10, dtype=torch.bool)
    mask[:6, :6] = True
    assert torch.equal(fragments.mask(), mask)
    assert len(fragments.pixels) == 36  # no pixel drawn twice along the diagonal


def test_the_nearer_triangle_wins():
    points = torch.cat((square(2, 6), square(1, 4)))
    faces = torch.cat((SQUARE, SQUARE + 4))
    fragments = rasterize(points, faces, 8, 8)
    depth = fragments.depth_image()
    assert (depth[:4, :4] == 1).all()
    assert (depth[4:6, :6] == 2).all()
    assert torch.isinf(depth[6:]).all()


def test_large_and_small_triangles_together():
    points = torch.cat((square(1, 300), square(0.5, 1.5)))
    fragments = rasterize(points, torch.cat((SQUARE, SQUARE + 4)), 320, 310)
    assert fragments.mask().sum() == 300 * 300
    assert (fragments.faces[fragments.pixels == 0] >= 2).all()


def test_interpolation_is_perspective_correct_and_differentiable():
    # A triangle receding in depth: its screen midpoint is nearer to the near corner in 3D.
    points = torch.tensor([[0.0, 0, 1], [64, 0, 3], [0, 64, 1]], requires_grad=True)
    faces = torch.tensor([[0, 1, 2]])
    fragments = rasterize(points, faces, 64, 64)
    assert_close(fragments.barycentric.sum(-1), torch.ones(len(fragments.pixels)))
    pixel = fragments.pixels == 0 * 64 + 31  # x = 31.5, half way across the top
    u = fragments.barycentric[pixel, 1]
    assert 0.1 < u.item() < 0.3  # a quarter of the way along in 3D, not a half
    depth = fragments.depth[pixel]
    assert_close(depth, 1 + 2 * u, atol=1e-4, rtol=0)
    depth.sum().backward()
    assert points.grad.abs().sum() > 0


def test_textures_are_sampled_with_v_up():
    texture = torch.zeros(2, 2, 3)
    texture[0, :, 0] = 1  # top row red
    colours = sample_texture(texture, torch.tensor([[0.5, 0.75], [0.5, 0.25]]))
    assert_close(colours[:, 0], torch.tensor([1.0, 0.0]))


def test_a_textured_mesh_hides_the_points_behind_it():
    camera = Camera.fixed(16, 16, 60)
    vertices = torch.tensor([[-1.0, 2, -1], [1, 2, -1], [1, 2, 1], [-1, 2, 1]])
    uvs = torch.tensor([[[0.0, 0], [1, 0], [1, 1]], [[0, 0], [1, 1], [0, 1]]])
    texture = torch.ones(4, 4, 3) * 0.5
    image = render_mesh(
        vertices, SQUARE, uvs, torch.zeros(2, dtype=torch.long), [texture], camera,
        torch.zeros(3),
    )  # fmt: skip
    assert image.mask[8, 8] == 1
    assert_close(image.image[8, 8], torch.full((3,), 0.5))
    points = torch.tensor([[0.0, 2.05, 0], [0, 2.5, 0], [0, 1.5, 0]])
    hidden = behind_mesh(points, torch.full((3,), 0.1), image.depth, camera)
    assert hidden.tolist() == [False, True, False]


def test_a_camera_looking_at_a_point_sees_it_in_the_middle():
    camera = Camera.fixed(64, 48, 40).looking_at(torch.tensor([1.0, -2, 0.5]), torch.zeros(3))
    assert_close(camera.project(torch.zeros(3)), torch.tensor([32.0, 24.0]))
    above = camera.project(torch.tensor([0.0, 0, 0.1]))
    assert above[1] < 24  # z is up in the image


def test_gaussians_whose_skin_faces_away_are_culled():
    camera = Camera.fixed(16, 16, 60)  # at the origin, looking along +y
    means = torch.tensor([[0.0, 2, 0], [0.0, 2, 0]])
    towards = torch.tensor([0.7071, 0.7071, 0, 0])  # normal (z) turned to -y, at the camera
    away = torch.tensor([0.7071, -0.7071, 0, 0])  # normal turned to +y
    assert facing_away(means, torch.stack((towards, away)), camera).tolist() == [False, True]
