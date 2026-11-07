from dataclasses import dataclass

import numpy as np
from numpy.testing import assert_array_equal

from avatar_splats.records import ArrayRecord


@dataclass(frozen=True)
class Example(ArrayRecord):
    values: np.ndarray
    names: np.ndarray
    count: int
    scale: float
    extra: np.ndarray | None = None


def test_round_trip_keeps_arrays_numbers_and_strings(tmp_path):
    record = Example(np.arange(6.0).reshape(2, 3), np.array(["jawOpen", "eyeBlink_L"]), 3, 0.5)
    record.save(tmp_path / "nested" / "example.npz")
    loaded = Example.load(tmp_path / "nested" / "example.npz")
    assert_array_equal(loaded.values, record.values)
    assert loaded.names.tolist() == ["jawOpen", "eyeBlink_L"]
    assert loaded.count == 3 and isinstance(loaded.count, int)
    assert loaded.scale == 0.5
    assert loaded.extra is None
