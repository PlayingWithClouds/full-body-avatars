import numpy as np
import pytest

from avatar_splats.data import probe, read_frames, write_video


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


def test_read_frames_start(synthetic_video):
    means = [frame.mean() for frame in read_frames(synthetic_video)]
    late = [frame.mean() for frame in read_frames(synthetic_video, step=2, start=5)]
    assert late == [means[5], means[7], means[9]]


def test_read_frames_rejects_bad_step(synthetic_video):
    with pytest.raises(ValueError):
        next(read_frames(synthetic_video, step=0))


def test_write_video_round_trip(tmp_path):
    frames = [np.full((48, 32, 3), 20 * i, np.uint8) for i in range(5)]
    write_video(tmp_path / "out.mp4", frames, fps=8)
    info = probe(tmp_path / "out.mp4")
    assert (info.width, info.height, info.num_frames) == (32, 48, 5)
    assert info.fps == pytest.approx(8)
    means = [frame.mean() for frame in read_frames(tmp_path / "out.mp4")]
    assert means == sorted(means)
