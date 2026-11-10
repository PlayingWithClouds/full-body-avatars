import math

import numpy as np
import torch
from torch.testing import assert_close

from avatar_splats.gaussians import BoundGaussians, GaussianRecord
from avatar_splats.training import ColourCorrection, fill_holes
from avatar_splats.training.textures import _bands, choose_views
from avatar_splats.training.train import mask_edge


def test_holes_are_filled_from_their_surroundings():
    weights = torch.zeros(8, 8)
    sums = torch.zeros(8, 8, 3)
    weights[:, :4] = 2
    sums[:, :4] = 2 * torch.tensor([0.2, 0.4, 0.6])
    weights[:, 4:6] = 1
    sums[:, 4:6] = torch.tensor([0.8, 0.8, 0.8])
    filled = fill_holes(sums, weights)
    assert_close(filled[:, 0], torch.tensor([0.2, 0.4, 0.6]).expand(8, 3))
    assert_close(filled[:, 5], torch.full((8, 3), 0.8))
    # The empty columns take a colour between their neighbours'.
    assert ((filled[:, 6:] >= 0.2) & (filled[:, 6:] <= 0.8)).all()


def test_colour_correction_leaves_the_reference_and_the_background_alone():
    colours = ColourCorrection(["body", "head"], reference="body")
    with torch.no_grad():
        colours.matrices["head"].mul_(2)
        colours.offsets["head"].fill_(0.1)
    colour = torch.full((2, 2, 3), 0.25)
    alpha = torch.tensor([[1.0, 0.5], [0.0, 1.0]])
    assert torch.equal(colours("body", colour * alpha[..., None], alpha), colour * alpha[..., None])
    corrected = colours("head", colour * alpha[..., None], alpha)
    assert_close(corrected[0, 0], torch.full((3,), 0.6))
    assert_close(corrected[1, 0], torch.zeros(3))  # nothing drawn, nothing added


def test_colours_turn_back_into_the_reference():
    colours = ColourCorrection(["body", "head"], reference="body")
    with torch.no_grad():
        colours.matrices["head"].copy_(
            torch.tensor([[1.1, 0.0, 0.0], [0.05, 0.9, 0.0], [0, 0, 1.2]])
        )
        colours.offsets["head"].fill_(-0.02)
    original = torch.rand(10, 3)
    shown = colours("head", original, torch.ones(10))
    assert_close(colours.to_reference("head", shown), original)


def test_the_edge_band_lies_on_both_sides_of_the_mask_edge():
    mask = torch.zeros(1, 20)
    mask[:, :10] = 1
    band = mask_edge(mask.expand(20, 20).contiguous(), 2)[10]
    assert band.nonzero().flatten().tolist() == [8, 9, 10, 11]


def test_scales_are_limited_in_size_and_elongation():
    scales = np.log([[1.0, 1.0, 0.1], [10.0, 1.0, 0.1], [0.5, 0.4, 0.1]], dtype=np.float32)
    gaussians = BoundGaussians(
        GaussianRecord(
            faces=np.zeros(3, np.int64),
            barycentric=np.full((3, 3), 1 / 3, np.float32),
            offsets=np.zeros((3, 3), np.float32),
            rotations=np.tile(np.float32([1, 0, 0, 0]), (3, 1)),
            log_scales=scales,
            opacity_logits=np.zeros(3, np.float32),
            sh=np.zeros((3, 1, 3), np.float32),
        )
    )
    gaussians.limit_scales(torch.full((3,), math.log(0.6)), max_elongation=4)
    assert_close(
        gaussians.log_scales.exp(),
        torch.tensor([[0.6, 0.6, 0.1], [0.6, 0.6, 0.1], [0.5, 0.4, 0.1]]),
    )
    gaussians.limit_scales(torch.full((3,), math.log(10.0)), max_elongation=4)
    assert_close(gaussians.log_scales.exp()[2], torch.tensor([0.5, 0.4, 0.1]))


def test_offsets_are_limited_in_length():
    gaussians = BoundGaussians(
        GaussianRecord(
            faces=np.zeros(2, np.int64),
            barycentric=np.full((2, 3), 1 / 3, np.float32),
            offsets=np.float32([[3, 0, 4], [0.1, 0, 0]]),
            rotations=np.tile(np.float32([1, 0, 0, 0]), (2, 1)),
            log_scales=np.zeros((2, 3), np.float32),
            opacity_logits=np.zeros(2, np.float32),
            sh=np.zeros((2, 1, 3), np.float32),
        )
    )
    gaussians.limit_offsets(torch.tensor([1.0, 1.0]))
    assert_close(gaussians.offsets, torch.tensor([[0.6, 0, 0.8], [0.1, 0, 0]]))


def test_bands_add_up_and_keep_the_background_out():
    image = torch.zeros(16, 16, 3)
    image[:, :8] = 0.6  # the performer on the left, a dark background on the right
    image[14, 4] = 1.0  # a fine detail, far from the rows checked at the edge
    mask = torch.zeros(16, 16)
    mask[:, :8] = 1
    broad, detail = _bands(image, mask, 2.0)
    assert_close(broad + detail, image)
    assert_close(broad[:6, 7], torch.full((6, 3), 0.6), atol=1e-6, rtol=0)  # edge stays skin
    assert detail[14, 4, 0] > 0.3 > broad[14, 4, 0] - 0.6


def test_triangles_take_their_neighbours_view_unless_their_own_is_much_better():
    quality = torch.tensor([
        [1.0, 0.5, 0.0],
        [1.0, 0.5, 0.0],
        [0.9, 1.0, 0.0],  # a little better seen by view 1, between two triangles on view 0
        [1.0, 0.2, 5.0],  # best seen by the fallback view, but a main view sees it too
        [0.0, 0.0, 1.0],  # seen by the fallback view only
        [0.0, 0.0, 0.0],  # seen by none
    ])  # fmt: skip
    fallback = torch.tensor([False, False, True])
    chain = torch.tensor([[0, 1], [1, 2], [2, 3]])

    assert choose_views(quality, fallback, chain, 0.25).tolist() == [0, 0, 0, 0, 2, -1]
    assert choose_views(quality, fallback, chain, 0.0).tolist() == [0, 0, 1, 0, 2, -1]
