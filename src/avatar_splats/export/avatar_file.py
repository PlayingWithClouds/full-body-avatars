"""The ``.avatar`` file: everything a game runtime needs to pose and draw the avatar.

Layout, little-endian:

- 8 bytes: the magic ``AVATAR01``
- 4 bytes: unsigned length of the header
- the header: UTF-8 JSON, padded with spaces to a multiple of 16 bytes
- the arrays, each starting at a multiple of 16 bytes from the start of the file

The header holds metadata and, for every array, its ``dtype``, ``shape`` and byte ``offset``.
Arrays are C-ordered. See ``runtime.py`` for how they are used.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from avatar_splats.body_models.hybrid import Avatar
from avatar_splats.gaussians import GaussianRecord

MAGIC = b"AVATAR01"
ALIGNMENT = 16


@dataclass(frozen=True)
class AvatarFile:
    metadata: dict[str, Any]
    arrays: dict[str, np.ndarray]


def avatar_arrays(avatar: Avatar, gaussians: GaussianRecord) -> AvatarFile:
    """Collect the avatar and its Gaussians into the file's arrays, in the runtime's dtypes."""
    quaternions = gaussians.rotations / np.linalg.norm(gaussians.rotations, axis=-1, keepdims=True)
    metadata = {
        "format_version": 1,
        "units": "metres",
        "up_axis": "+z",
        "forward_axis": "-y",
        "bones": avatar.bone_labels.tolist(),
        "expressions": avatar.expression_names.tolist(),
        "head_start": int(avatar.head_start),
        "sh_degree": int(np.sqrt(gaussians.sh.shape[1])) - 1,
        "quaternion_order": "wxyz",
    }
    arrays = {
        "rest_vertices": avatar.rest_vertices.astype(np.float32),
        "faces": avatar.faces.astype(np.uint32),
        "bone_indices": avatar.bone_indices.astype(np.uint16),
        "bone_weights": avatar.bone_weights.astype(np.float32),
        "bone_parents": avatar.bone_parents.astype(np.int32),
        "bind_poses": avatar.bind_poses.astype(np.float32),
        "expression_basis": avatar.expression_basis.astype(np.float32),
        "gaussian_faces": gaussians.faces.astype(np.uint32),
        "gaussian_barycentric": gaussians.barycentric.astype(np.float32),
        "gaussian_offsets": gaussians.offsets.astype(np.float32),
        "gaussian_rotations": quaternions.astype(np.float32),
        "gaussian_log_scales": gaussians.log_scales.astype(np.float32),
        "gaussian_opacities": (1 / (1 + np.exp(-gaussians.opacity_logits))).astype(np.float32),
        "gaussian_sh": gaussians.sh.astype(np.float32),
    }
    return AvatarFile(metadata, arrays)


def write_avatar(path: str | Path, file: AvatarFile) -> None:
    layout, offset = {}, 0
    for name, array in file.arrays.items():
        layout[name] = {"dtype": array.dtype.str, "shape": list(array.shape), "offset": offset}
        offset = _aligned(offset + array.nbytes)

    # The arrays start after the header, whose length depends on the offsets written into it.
    start, header = -1, b""
    while start != len(MAGIC) + 4 + len(header):
        start = len(MAGIC) + 4 + len(header)
        header = _header(file.metadata, layout, start)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as out:
        out.write(MAGIC + struct.pack("<I", len(header)) + header)
        for name, array in file.arrays.items():
            out.seek(layout[name]["offset"] + start)
            out.write(np.ascontiguousarray(array).tobytes())


def read_avatar(path: str | Path) -> AvatarFile:
    data = Path(path).read_bytes()
    if data[: len(MAGIC)] != MAGIC:
        raise ValueError(f"{path} is not an avatar file")
    (length,) = struct.unpack_from("<I", data, len(MAGIC))
    header = json.loads(data[len(MAGIC) + 4 : len(MAGIC) + 4 + length])
    arrays = {}
    for name, entry in header.pop("arrays").items():
        dtype = np.dtype(entry["dtype"])
        count = int(np.prod(entry["shape"]))
        array = np.frombuffer(data, dtype, count, entry["offset"])
        arrays[name] = array.reshape(entry["shape"]).copy()
    return AvatarFile(header, arrays)


def _header(metadata: dict[str, Any], layout: dict[str, dict], start: int) -> bytes:
    arrays = {name: {**entry, "offset": entry["offset"] + start} for name, entry in layout.items()}
    text = json.dumps({**metadata, "arrays": arrays}).encode()
    return text + b" " * (-(len(MAGIC) + 4 + len(text)) % ALIGNMENT)


def _aligned(offset: int) -> int:
    return offset + (-offset % ALIGNMENT)
