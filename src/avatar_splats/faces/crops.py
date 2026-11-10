"""The video's faces next to the avatar's: crops of every frame around the head, as one JPEG
atlas the viewer cuts its tiles from."""

from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from PIL import Image


def crop_box(outline: np.ndarray, margin: float) -> np.ndarray:
    """``(4,)`` the square x0, y0, x1, y1 around a head's ``(4,)`` outline (x0, y0, x1, y1),
    its side the outline's longer side times ``margin``, centred on it. It may reach beyond the
    image; the missing pixels are left black."""
    centre = (outline[:2] + outline[2:]) / 2
    half = margin * float((outline[2:] - outline[:2]).max()) / 2
    half = max(half, 8.0)
    return np.concatenate((centre - half, centre + half))


def atlas_shape(count: int) -> tuple[int, int]:
    """Columns and rows of an atlas of ``count`` square tiles, as square as it can be."""
    columns = max(1, math.ceil(math.sqrt(count)))
    return columns, max(1, math.ceil(count / columns))


def write_crops(
    path: Path,
    frames: Iterable[np.ndarray],
    boxes: np.ndarray,
    tile: int,
    quality: int,
) -> tuple[int, int]:
    """Write the crops of ``frames`` as ``tile``-pixel squares in a JPEG atlas, row by row.

    Args:
        frames: RGB ``uint8`` frames ``(H, W, 3)``.
        boxes: ``(N, 4)`` each frame's crop, as ``crop_box`` gives it.

    Returns:
        The atlas's columns and rows.
    """
    columns, rows = atlas_shape(len(boxes))
    atlas = Image.new("RGB", (columns * tile, rows * tile))
    for i, (frame, box) in enumerate(zip(frames, boxes, strict=True)):
        left, top, right, bottom = (round(float(v)) for v in box)
        crop = Image.fromarray(frame).crop((left, top, right, bottom))  # black beyond the frame
        crop = crop.resize((tile, tile), Image.Resampling.LANCZOS)
        atlas.paste(crop, ((i % columns) * tile, (i // columns) * tile))
    path.parent.mkdir(parents=True, exist_ok=True)
    atlas.save(path, quality=quality)
    return columns, rows
