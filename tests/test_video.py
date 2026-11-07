import numpy as np
import pytest

from avatar_splats.data import probe, read_frames


def test_probe(synthetic_video):
    info = probe(synthetic_video)
    assert (info.width, info.height) == (32, 48)
    assert info.fps == pytest.approx(10)
    assert info.num_frames == 10
    assert info.duration == pytest.approx(1)


def test_read_frames_shape_and_order(synthetic_video):
    frames = list(read_frames(synthetic_video))
    assert len(frames) == 10
    assert frames[0].shape == (48, 32, 3)
    assert frames[0].dtype == np.uint8
    means = [frame.mean() for frame in frames]
    assert means == sorted(means)


def test_read_frames_step_and_limit(synthetic_video):
    assert len(list(read_frames(synthetic_video, step=3))) == 4
    assert len(list(read_frames(synthetic_video, step=2, max_frames=3))) == 3


def test_read_frames_rejects_bad_step(synthetic_video):
    with pytest.raises(ValueError):
        next(read_frames(synthetic_video, step=0))
