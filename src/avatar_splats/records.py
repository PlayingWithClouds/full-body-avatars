"""Stage results as dataclasses of arrays, stored as ``.npz`` files."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Self

import numpy as np


class ArrayRecord:
    """Mixin for dataclasses whose fields are arrays, numbers or strings.

    Fields that are None are not stored and load as their default.
    """

    def save(self, path: str | Path) -> None:
        fields = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{k: v for k, v in fields.items() if v is not None})

    @classmethod
    def load(cls, path: str | Path) -> Self:
        with np.load(path) as data:
            return cls(**{name: _unpack(data[name]) for name in data.files})


def _unpack(array: np.ndarray):
    return array.item() if array.ndim == 0 else array
