import cv2
import numpy as np
import pytest
import torch
from numpy.testing import assert_array_equal

from avatar_splats.preprocess import Preprocessed
from avatar_splats.preprocess.keypoints import coco_keypoints, extra_keypoints, hand_landmarks
from avatar_splats.preprocess.prepare import (
    box_iou,
    face_box,
    follow,
    outline,
    person_crop,
    zoom_cuts,
)
from avatar_splats.preprocess.segmentation import mask_boxes
from avatar_splats.preprocess.triage import context_crop, padded_crop, rank_shots, shot_pieces


def test_mask_boxes():
    masks = np.zeros((2, 10, 8), dtype=bool)
    masks[0, 2:5, 3:7] = True
    boxes = mask_boxes(masks)
    assert_array_equal(boxes[0], [3, 2, 7, 5])
    assert np.isnan(boxes[1]).all()


def test_preprocessed_round_trip(tmp_path):
    result = Preprocessed(
        frame_indices=np.array([0, 2]),
        masks=np.ones((2, 4, 3), dtype=bool),
        boxes=np.array([[0, 0, 3, 4], [np.nan] * 4]),
        keypoints=np.random.default_rng(0).random((2, 17, 3)),
    )
    path = tmp_path / "out" / "body.npz"
    result.save(path)
    loaded = Preprocessed.load(path)

    assert loaded.face_landmarks is None
    for name in ("frame_indices", "masks", "boxes", "keypoints"):
        assert_array_equal(getattr(loaded, name), getattr(result, name))


def test_rank_shots():
    times = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
    shot = np.array([0, 0, 1, 1, 1])
    score = np.array([0.2, 0.4, 0.9, 0.5, 0.7])
    best, worst = rank_shots(times, shot, score, end=9.5)
    assert (best.begin, best.end, best.best, best.score) == (4.0, 9.5, 2, 0.7)
    assert_array_equal(best.samples, [2, 3, 4])
    assert (worst.begin, worst.end, worst.best) == (0.0, 4.0, 1)


def test_shot_pieces_cut_shots_and_pick_their_best_sample():
    times = np.array([0.0, 1.0, 2.0, 3.0, 10.0])
    score = np.array([0.5, 0.9, 0.4, 0.6, 0.8])
    shots = rank_shots(times, np.array([0, 0, 0, 0, 1]), score, end=10.5)
    # fps 10: shot 0 spans frames 0-99, shot 1 frames 100-104
    pieces = shot_pieces(shots, times, score, fps=10, frame_step=2, max_frames=20, min_frames=3)
    first = [p for p in pieces if p.first < 100]
    assert [p.first for p in first] == [0, 40, 80]
    assert [p.count for p in first] == [20, 20, 10]
    assert first[0].select == 5  # the sample at 1 s: frame 10, the 5th of every 2nd
    assert first[2].select == 5  # no sample inside: the middle
    assert [(p.first, p.count) for p in pieces if p.first >= 100] == [(100, 3)]


def test_shot_pieces_drop_short_tails():
    shots = rank_shots(np.array([0.0]), np.array([0]), np.array([1.0]), end=2.5)
    pieces = shot_pieces(shots, np.array([0.0]), np.array([1.0]), 10, 1, 20, 8)
    assert [(p.first, p.count) for p in pieces] == [(0, 20)]  # the last 5 frames are dropped


def test_crops():
    image = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    crop = padded_crop(image, [-1, 1, 3, 3], fill=(7, 7, 7))
    assert crop.shape == (2, 4, 3)
    assert (crop[:, 0] == 7).all()
    assert (crop[:, 1:] == image[1:3, 0:3]).all()
    x0, y0, x1, y1 = context_crop(np.array([0, 0, 30, 40]), 2.0)
    assert (x1 - x0, y1 - y0) == (60, 80) and (x0 + x1, y0 + y1) == (30, 40)
    x0, y0, x1, y1 = person_crop(np.array([10.0, 0.0, 20.0, 100.0]), 0.0)
    assert (y1 - y0) == 100 and abs((x1 - x0) - 75) <= 1  # 3:4


def test_face_box_needs_three_confident_points():
    points = np.zeros((17, 3))
    points[:5, :2] = [[50, 50], [45, 45], [55, 45], [40, 48], [60, 48]]
    points[:5, 2] = [0.9, 0.9, 0.9, 0.1, 0.1]
    x0, y0, x1, y1 = face_box(points, 1.0, 0.3, (200, 200))
    assert x0 < 45 and x1 > 55 and y0 < 45 and y1 > 50
    points[1, 2] = 0.1
    assert face_box(points, 1.0, 0.3, (200, 200)) is None


def test_follow_keeps_to_the_overlapping_box():
    me = np.array([[10.0, 10, 50, 90, 0.9]])
    other = np.array([[60.0, 10, 70, 30, 0.95]])
    found = [
        np.concatenate([other, me + [2, 0, 2, 0, 0]]),
        np.concatenate([other, me]),  # select: I am the larger
        np.zeros((0, 5)),  # nobody found
        np.concatenate([other, me + [4, 0, 4, 0, 0]]),
        other,  # only the other one: I am lost
    ]
    followed = follow(found, select=1, min_iou=0.5)
    assert followed[0, 0] == 12 and followed[1, 0] == 10 and followed[3, 0] == 14
    assert np.isnan(followed[[2, 4]]).all()


def test_box_iou():
    boxes = np.array([[0.0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]])
    np.testing.assert_allclose(box_iou(boxes[0], boxes), [1, 1 / 3, 0])


def test_outline_keeps_regions_mostly_inside_the_box():
    body = np.zeros((20, 30), bool)
    body[2:18, 2:10] = True  # inside
    body[5:8, 9:20] = True  # an arm reaching out, joined to it
    body[2:18, 24:29] = True  # someone else
    kept = outline(body, np.array([2, 2, 10, 18]), 0.0)
    assert kept[10, 5] and kept[6, 18] and not kept[10, 26]


def test_person_crops_match_vitposes_processor():
    from transformers import AutoProcessor

    from avatar_splats.preprocess.keypoints import person_crops

    processor = AutoProcessor.from_pretrained("usyd-community/vitpose-base-simple")
    rng = np.random.default_rng(4)
    image = (rng.random((120, 160, 3)) * 255).astype(np.uint8)
    image = np.ascontiguousarray(np.repeat(np.repeat(image, 4, 0), 4, 1))  # smooth-ish, 480x640
    # The processor rounds its warped crop to integers before normalizing. Where the crop leaves
    # the image its SciPy warp blackens a strip cv2's (and ours) samples, so that box is only
    # compared on average.
    for box, max_difference in (
        ([100.0, 50.0, 120.0, 300.0], 0.05),
        ([-20.0, 300.0, 200.0, 250.0], None),
    ):
        expected = processor(images=[image], boxes=[[box]], return_tensors="pt").pixel_values[0]
        crop = person_crops([torch.from_numpy(image)], [box], processor)[0]
        difference = (crop - expected).abs()
        assert difference.mean() < 0.02, box
        if max_difference is not None:
            assert difference.max() < max_difference, box


def test_padded_on_keeps_the_frame_inside_and_black_outside():
    from avatar_splats.preprocess.prepare import padded_on

    frame = np.arange(5 * 6 * 3, dtype=np.uint8).reshape(5, 6, 3)
    crop = padded_on(frame, (-2, 1, 3, 4), "cpu")  # (3, 3, 5): two columns outside
    assert crop.shape == (3, 3, 5)
    assert (crop[:, :, :2] == 0).all()
    assert (crop.permute(1, 2, 0).numpy()[:, 2:] == frame[1:4, 0:3]).all()


def test_focus_prefers_detail():
    from avatar_splats.preprocess.triage import focus

    rng = np.random.default_rng(5)
    sharp = (rng.random((200, 100, 3)) * 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(sharp, (15, 15), 5)
    box = np.array([10.0, 10.0, 90.0, 190.0])
    assert focus(sharp, box) > 10 * focus(blurred, box)
    assert np.isnan(focus(sharp, np.full(4, np.nan)))


def test_focus_is_not_fooled_by_a_soft_background():
    from avatar_splats.preprocess.triage import focus

    rng = np.random.default_rng(7)
    sharp = (rng.random((256, 256, 3)) * 255).astype(np.uint8)
    bokeh = cv2.GaussianBlur(sharp, (31, 31), 10)
    bokeh[64:192, 96:160] = sharp[64:192, 96:160]  # the person in focus, a quarter of the box
    box = np.array([0.0, 0.0, 256.0, 256.0])
    person = np.zeros((256, 256), bool)
    person[64:192, 96:160] = True
    assert focus(bokeh, box) > 0.5 * focus(sharp, box)
    assert focus(bokeh, box, person) == pytest.approx(focus(sharp, box, person), rel=0.05)
    blurred = cv2.GaussianBlur(sharp, (15, 15), 5)
    assert focus(sharp, box, person) > 10 * focus(blurred, box, person)


def test_good_stretches_split_a_long_shot():
    from avatar_splats.preprocess.triage import good_stretches

    times = np.arange(12, dtype=float) * 2
    shot = np.array([0] * 10 + [1] * 2)
    score = np.array([0.0, 0.95, 0.95, 0.2, 0.95, 0.0, 0.0, 0.92, 0.93, 0.0, 0.99, 0.99])
    stretches = good_stretches(times, shot, score, end=24.0, min_score=0.9, max_dip=1)
    spans = sorted((s.begin, s.end, tuple(s.samples)) for s in stretches)
    assert spans == [
        (2.0, 10.0, (1, 2, 3, 4)),  # one dip bridged, the trailing bad ones trimmed
        (14.0, 18.0, (7, 8)),
        (20.0, 24.0, (10, 11)),  # the second shot, to the video's end
    ]
    assert [s.score for s in stretches] == pytest.approx([0.99, 0.95, 0.925])


def test_skin_coloured_frees_clothing_that_is_skin():
    from avatar_splats.preprocess.prepare import PART_NAMES, skin_coloured

    rng = np.random.default_rng(6)
    frame = np.zeros((60, 90, 3), np.uint8)
    labels = np.zeros((60, 90), np.uint8)
    skin_colour, cloth_colour = np.array([210, 160, 130]), np.array([40, 60, 160])
    frame[:, :30] = skin_colour + rng.integers(-6, 7, (60, 30, 3))
    labels[:, :30] = PART_NAMES.index("Torso")
    frame[:, 30:60] = np.clip(skin_colour * 0.7 + rng.integers(-6, 7, (60, 30, 3)), 0, 255)
    frame[:, 62:] = cloth_colour + rng.integers(-6, 7, (60, 28, 3))  # a gap: two regions
    clothing = np.zeros((60, 90), bool)
    clothing[:, 30:60] = clothing[:, 62:] = True
    labels[clothing] = PART_NAMES.index("Lower_Clothing")
    freed = skin_coloured(frame, labels, clothing, 3.0)
    assert freed[:, 30:60].all()  # darker, but the skin's colour: a bare leg in shade
    assert not freed[:, 62:].any()  # blue: clothing


def test_zoom_cuts_at_jumps_in_face_size():
    face = np.array([[0, 0], [-3, -2], [3, -2], [-6, 0], [6, 0]], float)
    keypoints = np.zeros((5, 17, 3))
    for i, size in enumerate([1.0, 1.2, 1.2, 3.5, 3.4]):
        keypoints[i, :5, :2] = 300 + size * face
        keypoints[i, :5, 2] = 0.9
    keypoints[2, :5, 2] = 0.1  # no face: compared with the one before
    assert_array_equal(zoom_cuts(keypoints, 0.3), [False, False, False, True, False])


def test_goliath_keypoints_in_our_orders():
    goliath = np.zeros((2, 308, 3))
    goliath[..., 0] = np.arange(308)  # each point's x is its Goliath index
    coco = coco_keypoints(goliath)[0, :, 0]
    assert_array_equal(coco[[9, 10, 11, 16]], [62, 41, 9, 14])  # wrists, left hip, right ankle
    assert_array_equal(extra_keypoints(goliath)[0, [0, 2, 6, 11], 0], [15, 17, 63, 68])
    hands = hand_landmarks(goliath)[0, :, :, 0]
    assert hands.shape == (2, 21)
    assert_array_equal(hands[0, [0, 1, 4, 5, 8, 20]], [62, 45, 42, 49, 46, 58])  # left hand
    assert_array_equal(hands[1, [0, 1, 4]], [41, 24, 21])  # right: wrist, thumb base, tip


def test_is_prepared_rejects_other_stretches_and_versions(tmp_path):
    from dataclasses import fields

    from avatar_splats.preprocess.prepare import PreparedShot, is_prepared

    def piece(**changes):
        values = {field.name: np.zeros(1) for field in fields(PreparedShot)}
        values["frame_indices"] = np.arange(100, 130, 5)
        values.update(changes)
        return values

    path = tmp_path / "piece.npz"
    assert not is_prepared(path, 100, 5, 6)
    np.savez(path, **piece())
    assert is_prepared(path, 100, 5, 6)
    assert is_prepared(path, 100, 5, 8)  # fewer frames: the video ended
    assert not is_prepared(path, 105, 5, 6)  # another stretch
    assert not is_prepared(path, 100, 2, 6)  # another step
    assert not is_prepared(path, 100, 5, 4)
    np.savez(path, **{k: v for k, v in piece().items() if k != "sharpness"})
    assert not is_prepared(path, 100, 5, 6)  # an older version's
    np.savez(path, **piece(hmr_joints=np.zeros(1)))
    assert not is_prepared(path, 100, 5, 6)


def test_boxed_greys_everything_outside_the_box():
    from avatar_splats.preprocess.prepare import GREY, boxed

    frame = np.full((40, 60, 3), 200, np.uint8)
    out = boxed(frame, np.array([10.0, 5.0, 30.0, 25.0]), margin=0.1)  # widened by 2 px
    assert (out[3:27, 8:32] == 200).all()
    assert (out[0, 0] == GREY).all() and (out[30, 40] == GREY).all() and (out[2, 20] == GREY).all()
    assert (boxed(frame, np.full(4, np.nan), 0.1) == GREY).all()


def test_covered_share_of_her_box():
    from avatar_splats.preprocess.prepare import covered_share

    hers = np.array([0.0, 0.0, 10.0, 10.0])
    alone = np.array([[0.0, 0.0, 10.0, 10.0, 0.9]])
    interlocked = np.array([[0.0, 0.0, 10.0, 10.0, 0.9], [5.0, 0.0, 15.0, 10.0, 0.8]])
    assert covered_share(hers, alone) == 0
    assert covered_share(hers, interlocked) == pytest.approx(0.5)
    assert np.isnan(covered_share(np.full(4, np.nan), interlocked))


def test_focus_ignores_contrast():
    from avatar_splats.preprocess.triage import focus

    rng = np.random.default_rng(8)
    sharp = cv2.GaussianBlur((rng.random((256, 256, 3)) * 255).astype(np.uint8), (3, 3), 0.7)
    faint = (128 + (sharp.astype(float) - 128) * 0.2).astype(np.uint8)  # smooth skin's contrast
    box = np.array([0.0, 0.0, 256.0, 256.0])
    assert focus(faint, box) == pytest.approx(focus(sharp, box), rel=0.25)
    assert focus(faint, box) > 3 * focus(cv2.GaussianBlur(sharp, (9, 9), 3), box)


def test_skin_coloured_keeps_a_print_of_the_skins_hue():
    from avatar_splats.preprocess.prepare import PART_NAMES, skin_coloured

    rng = np.random.default_rng(9)
    skin_colour = np.array([200, 150, 120])
    frame = np.clip(skin_colour + rng.normal(0, 3, (80, 120, 3)), 0, 255).astype(np.uint8)
    labels = np.full((80, 120), PART_NAMES.index("Torso"), np.uint8)
    regions = np.zeros((80, 120), bool)
    regions[10:70, 70:110] = True  # skin labelled as clothing: smooth, the skin's colour
    labels[regions] = PART_NAMES.index("Lower_Clothing")
    assert skin_coloured(frame, labels, regions, 3.0)[regions].all()
    # a print: the skin's hue, with dark spots every few pixels
    spots = (np.add.outer(np.arange(80) // 3, np.arange(120) // 3) % 3 == 0) & regions
    frame[spots] = (40, 25, 15)
    assert not skin_coloured(frame, labels, regions, 3.0).any()
