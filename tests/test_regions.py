import numpy as np
import pytest

from avatar_splats.preprocess.prepare import NOT_THEM, PART_NAMES
from avatar_splats.preprocess.regions import (
    RegionQualitySettings,
    clothed_share,
    detections_of,
    detections_table,
    measure_regions,
    region_parts,
    score_regions,
    shot_identity,
    worth_segmenting,
)

NAMES = {0: "FEMALE_BREAST_EXPOSED", 1: "FEMALE_GENITALIA_EXPOSED", 2: "HAND"}


def settings(**changes) -> RegionQualitySettings:
    values = dict(
        edge=0.01, good_pixels=200.0, good_focus=0.05, dark=0.2, bright=0.5, max_clipped=0.2,
        uneven=0.3, even=0.7, max_occlusion=0.3, max_clothed=0.4, max_overlap=0.3,
        other_face=0.2, same_face=0.4,
        weights={"resolution": 1.0, "occlusion": 1.0},
        gates={"found": 1.0, "occlusion": 0.5, "bare": 0.5, "resolution": 0.5}, min_score=0.3,
    )  # fmt: skip
    values.update(changes)
    return RegionQualitySettings(**values)


def test_detections_table_round_trip():
    found = [np.zeros((0, 6)), np.array([[1, 2, 3, 4, 0.9, 0], [5, 6, 7, 8, 0.8, 2]])]
    back = detections_of(detections_table(found), 3)
    assert len(back) == 3 and len(back[0]) == 0 and len(back[2]) == 0
    assert np.allclose(back[1], found[1])


def test_region_parts_are_hers_and_the_most_confident():
    detections = np.array([
        [10, 10, 20, 20, 0.6, 0], [30, 10, 40, 20, 0.9, 0], [50, 10, 60, 20, 0.8, 0],
        [200, 10, 210, 20, 0.95, 0],  # someone else's
        [30, 40, 40, 50, 0.7, 1],
    ])  # fmt: skip
    person = np.array([0.0, 0, 100, 100])
    breasts = region_parts(detections, NAMES, "breasts", person)
    assert breasts[:, 4].tolist() == [0.9, 0.8]
    assert len(region_parts(detections, NAMES, "vulva", person)) == 1
    assert region_parts(detections, NAMES, "breasts", np.full(4, np.nan))[0, 4] == 0.95


def test_shot_identity_fills_unseen_faces_from_their_shot():
    identity = np.array([0.5, np.nan, 0.3, np.nan, np.nan])
    shot = np.array([0, 0, 0, 1, 1])
    filled = shot_identity(identity, shot)
    assert filled[1] == pytest.approx(0.4) and filled[0] == 0.5
    assert np.isnan(filled[3:]).all()


def test_regions_measured_and_hands_over_them():
    frame = np.full((400, 400, 3), 170, np.uint8)
    breasts = [[100, 150, 180, 230, 0.9, 0], [200, 150, 280, 230, 0.9, 0]]
    hand = [[100, 150, 180, 230, 0.8, 2]]  # over one breast
    detections = [np.array(breasts), np.array(breasts + hand), np.zeros((0, 6))]
    measures = measure_regions(
        [frame] * 3, detections, NAMES, "breasts", None, 0.5, np.zeros(3), settings()
    )
    assert measures["found"].tolist() == [True, True, False]
    assert measures["pixels"][0] == pytest.approx(360) and measures["parts"][0] == 2
    assert measures["occlusion"][0] == 0 and measures["occlusion"][1] == pytest.approx(0.5, 0.05)
    assert measures["brightness"][0] == pytest.approx(170 / 255, abs=0.01)
    _, _, passed, reasons = score_regions(measures, settings())
    assert passed[0] and not passed[1] and "occlusion" in reasons[1]
    assert reasons[2].startswith("found")


def test_clothes_over_a_region_fail_it_where_segmented():
    labels = np.full((400, 400), NOT_THEM, np.uint8)
    labels[100:300, 100:300] = PART_NAMES.index("Torso")
    clothed = labels.copy()
    clothed[100:300, 100:300] = PART_NAMES.index("Lower_Clothing")
    box = np.array([[150.0, 150, 250, 250, 0.9]])
    assert clothed_share(labels, box) == 0 and clothed_share(clothed, box) == 1
    assert np.isnan(clothed_share(np.full((400, 400), NOT_THEM, np.uint8), box))
    frame = np.full((400, 400, 3), 170, np.uint8)
    detections = [np.array([[150, 150, 250, 250, 0.9, 1]])] * 3
    measures = measure_regions(
        [frame] * 3, detections, NAMES, "vulva", None, 1.0, np.zeros(3), settings(),
        labels=[labels, clothed, None],
    )  # fmt: skip
    factors, _, passed, reasons = score_regions(measures, settings())
    assert factors["bare"].tolist() == [1.0, 0.0, 1.0]
    assert passed[0] and not passed[1] and reasons[1].startswith("bare") and passed[2]


def test_only_regions_big_enough_are_worth_segmenting():
    detections = [np.array([[0, 0, 150, 150, 0.9, 1]]), np.array([[0, 0, 50, 50, 0.9, 1]]),
                  np.array([[0, 0, 150, 150, 0.9, 2]])]  # fmt: skip
    worth = worth_segmenting(detections, NAMES, None, 1.0, {"vulva": settings()})
    assert worth.tolist() == [True, False, False]
