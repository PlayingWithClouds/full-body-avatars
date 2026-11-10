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

import dataclasses
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from avatar_splats.body_models.hybrid import EYES, MOUTH, PART_NAMES, Avatar
from avatar_splats.export.face_hair import FaceHair
from avatar_splats.export.mouth import mouth_metadata, mouth_regions
from avatar_splats.gaussians import GaussianRecord
from avatar_splats.training.textures import TextureRecord

MAGIC = b"AVATAR01"
ALIGNMENT = 16


@dataclass(frozen=True)
class AvatarFile:
    metadata: dict[str, Any]
    arrays: dict[str, np.ndarray]


def avatar_arrays(
    avatar: Avatar,
    gaussians: GaussianRecord,
    textures: TextureRecord,
    occlusion_margin: float,
    eyes: dict[str, Any],
    face_hair: FaceHair,
    face_hair_entry: dict[str, Any],
) -> AvatarFile:
    """Collect the avatar, its textures and its Gaussians into the file's arrays, in the
    runtime's dtypes. Textures are 8-bit RGB, their first row at the top (v = 1).

    Every triangle's part is stored, named by ``parts``: the runtime draws the ``eyes`` with
    their own material (see ``eyes.py``), from the ``eyes`` entry, not from the textures, and
    the Gaussians trained on them are left out; so does the inside of the mouth, by the regions
    in ``mouth_regions`` and the ``mouth`` entry's colours (see ``mouth.py``). The lashes and
    brows are strands bound to triangles (see ``face_hair.py``), named by the ``face_hair``
    entry's groups.

    Args:
        occlusion_margin: how far, in sizes of its triangle, a Gaussian may lie behind the mesh
            and still be drawn; the margin training used.
        eyes: the eyes' metadata from ``eye_metadata``.
        face_hair_entry: the face hair's metadata from ``face_hair_metadata``.
    """
    gaussians = _without(gaussians, np.isin(avatar.face_parts[gaussians.faces], (EYES, MOUTH)))
    quaternions = gaussians.rotations / np.linalg.norm(gaussians.rotations, axis=-1, keepdims=True)
    metadata = {
        "format_version": 3,
        "units": "metres",
        "up_axis": "+z",
        "forward_axis": "-y",
        "bones": avatar.bone_labels.tolist(),
        "expressions": avatar.expression_names.tolist(),
        "head_start": int(avatar.head_start),
        "sh_degree": int(np.sqrt(gaussians.sh.shape[1])) - 1,
        "quaternion_order": "wxyz",
        "textures": ["body_texture", "head_texture"],
        "occlusion_margin": occlusion_margin,
        "parts": list(PART_NAMES),
        "eyes": eyes,
        "face_hair": face_hair_entry,
        "mouth": mouth_metadata(avatar, textures),
    }
    arrays = {
        "rest_vertices": avatar.rest_vertices.astype(np.float32),
        "faces": avatar.faces.astype(np.uint32),
        "uvs": avatar.uvs.astype(np.float32),
        "face_textures": avatar.face_textures.astype(np.uint8),
        "face_parts": avatar.face_parts.astype(np.uint8),
        "mouth_regions": mouth_regions(avatar),
        "body_texture": _bytes(textures.body),
        "head_texture": _bytes(textures.head),
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
        "hair_faces": face_hair.faces.astype(np.uint32),
        "hair_barycentric": face_hair.barycentric.astype(np.float32),
        "hair_points": face_hair.points.astype(np.float32),
        "hair_widths": face_hair.widths.astype(np.float32),
        "hair_groups": face_hair.groups.astype(np.uint8),
    }
    return AvatarFile(metadata, arrays)


def _without(gaussians: GaussianRecord, dropped: np.ndarray) -> GaussianRecord:
    fields = dataclasses.fields(gaussians)
    return dataclasses.replace(
        gaussians, **{field.name: getattr(gaussians, field.name)[~dropped] for field in fields}
    )


def _bytes(texture: np.ndarray) -> np.ndarray:
    return np.round(np.clip(texture, 0, 1) * 255).astype(np.uint8)


def write_avatar(path: str | Path, file: AvatarFile, magic: bytes = MAGIC) -> None:
    """Write ``file`` in the layout above; other files in the same layout (``.motion``) give
    their own 8-byte ``magic``."""
    layout, offset = {}, 0
    for name, array in file.arrays.items():
        layout[name] = {"dtype": array.dtype.str, "shape": list(array.shape), "offset": offset}
        offset = _aligned(offset + array.nbytes)

    # The arrays start after the header, whose length depends on the offsets written into it.
    start, header = -1, b""
    while start != len(magic) + 4 + len(header):
        start = len(magic) + 4 + len(header)
        header = _header(file.metadata, layout, start)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as out:
        out.write(magic + struct.pack("<I", len(header)) + header)
        for name, array in file.arrays.items():
            out.seek(layout[name]["offset"] + start)
            out.write(np.ascontiguousarray(array).tobytes())


def read_avatar(path: str | Path, magic: bytes = MAGIC) -> AvatarFile:
    data = Path(path).read_bytes()
    if data[: len(magic)] != magic:
        raise ValueError(f"{path} does not start with {magic.decode()}")
    (length,) = struct.unpack_from("<I", data, len(magic))
    header = json.loads(data[len(magic) + 4 : len(magic) + 4 + length])
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
