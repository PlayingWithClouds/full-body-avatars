import numpy as np
from numpy.testing import assert_array_equal

from avatar_splats.preprocess import Preprocessed
from avatar_splats.preprocess.segmentation import mask_boxes


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
