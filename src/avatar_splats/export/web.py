"""A small ``.avatar`` for the web: the textured mesh only, for a page that shows the avatar.

From an exported avatar file, ``web_avatar`` keeps what a page draws and drops the rest:

- The genitals are smoothed away (``smooth_crotch``): the canals are dropped, the generated
  surface becomes a smooth membrane between the edges it was stitched to, and the canals'
  openings are closed. Their blendshapes go too.
- Only the expressions a page animates are kept (``WebConfig.expressions``): the blendshapes
  are most of the file.
- The Gaussians are dropped; the textures are written beside the file as WebP images, named by
  the ``web_textures`` metadata entry (paths relative to the file), instead of raw arrays.

The result is read like any ``.avatar`` file, its textures loaded from the images.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_linalg
from PIL import Image

from avatar_splats.config import Config
from avatar_splats.export.avatar_file import AvatarFile, read_avatar, write_avatar

FACE_ARRAYS = ("faces", "uvs", "face_textures", "face_parts", "mouth_regions")
"""Arrays with one row per triangle, in the triangles' order."""


def export_web(config: Config, name: str) -> dict[str, Path]:
    """Write ``outputs/web/<name>/``: the web file of ``outputs/export/<name>.avatar`` and its
    texture images."""
    source = config.outputs / "export" / f"{name}.avatar"
    directory = config.outputs / "web" / name
    directory.mkdir(parents=True, exist_ok=True)
    file, images = web_avatar(read_avatar(source), config.web.expressions)
    written = {}
    for texture, image in images.items():
        path = directory / f"{texture}.webp"
        Image.fromarray(image).save(path, quality=config.web.texture_quality, method=6)
        written[texture] = path
    path = directory / f"{name}.avatar"
    write_avatar(path, file)
    written["avatar"] = path
    return written


def web_avatar(
    file: AvatarFile, expressions: list[str]
) -> tuple[AvatarFile, dict[str, np.ndarray]]:
    """The web version of ``file``, keeping the named ``expressions``, and its ``(H, W, 3)``
    8-bit textures by name, first row at the top."""
    metadata = dict(file.metadata)
    if unknown := set(expressions) - set(metadata["expressions"]):
        raise ValueError(f"the avatar has no expressions {sorted(unknown)}")
    arrays = dict(file.arrays)
    for gaussian in [key for key in arrays if key.startswith("gaussian_")]:
        arrays[gaussian] = arrays[gaussian][:0]
    arrays = smooth_crotch(arrays, metadata["parts"])

    kept = [metadata["expressions"].index(e) for e in expressions]
    arrays["expression_basis"] = np.ascontiguousarray(arrays["expression_basis"][kept])
    metadata["expressions"] = list(expressions)
    metadata.pop("wrinkles", None)
    arrays.pop("wrinkle_textures", None)

    images = {texture: arrays.pop(texture) for texture in metadata["textures"]}
    metadata["web_textures"] = {texture: f"{texture}.webp" for texture in images}
    return AvatarFile(metadata, arrays), images


def smooth_crotch(arrays: dict[str, np.ndarray], parts: list[str]) -> dict[str, np.ndarray]:
    """Replace the generated genitals with a smooth surface.

    The canals' triangles are dropped. The genitals' vertices that no other part uses move to
    the surface of least bending (a biharmonic fill) between those that it does, and each
    opening the canals leave is closed by a fan around a new vertex at its centre, skinned
    like the opening's rim. The triangles' order is kept otherwise, so ``hair_faces`` are
    renumbered; vertices are only added, so vertex indices hold.
    """
    faces = arrays["faces"].astype(np.int64)
    part = arrays["face_parts"]
    genitals, canals = parts.index("genitals"), parts.index("canals")
    surface = faces[part == genitals]
    if not len(surface):
        # Exported before the genitals were generated: the body is closed already.
        return arrays
    elsewhere = faces[(part != genitals) & (part != canals)]
    vertices = arrays["rest_vertices"].astype(np.float64)

    pinned = np.zeros(len(vertices), bool)
    pinned[elsewhere.ravel()] = True
    free = np.setdiff1d(np.unique(surface), np.flatnonzero(pinned))
    laplacian = _laplacian(surface, len(vertices))
    bending = (laplacian @ laplacian).tocsr()
    rest = np.setdiff1d(np.arange(len(vertices)), free)
    system = sparse_linalg.splu(bending[free][:, free].tocsc())
    vertices[free] = system.solve(-(bending[free][:, rest] @ vertices[rest]))

    openings = _openings(surface, pinned)
    # Wound like the surface: each rim edge runs the other way in the triangle it borders.
    openings = [loop[::-1] if _shares_direction(surface, loop[:2]) else loop for loop in openings]
    centres = np.array([vertices[loop].mean(axis=0) for loop in openings]).reshape(-1, 3)
    first = len(vertices)
    fan = np.array(
        [
            [loop[i], loop[(i + 1) % len(loop)], first + k]
            for k, loop in enumerate(openings)
            for i in range(len(loop))
        ],
        dtype=np.int64,
    ).reshape(-1, 3)

    out = dict(arrays)
    out["rest_vertices"] = np.concatenate([vertices, centres]).astype(np.float32)
    rim = np.array([loop[0] for loop in openings], dtype=np.int64)
    for name in ("bone_indices", "bone_weights"):
        out[name] = np.concatenate([arrays[name], arrays[name][rim]])
    basis = arrays["expression_basis"]
    out["expression_basis"] = np.concatenate(
        [basis, np.zeros((basis.shape[0], len(openings), 3), basis.dtype)], axis=1
    )

    keep = part != canals
    renumber = np.cumsum(keep) - 1
    fan_rows = {
        "faces": fan.astype(arrays["faces"].dtype),
        "uvs": np.repeat(arrays["uvs"][part == genitals][:1], len(fan), axis=0),
        "face_textures": np.full(len(fan), arrays["face_textures"][part == genitals][0]),
        "face_parts": np.full(len(fan), genitals, arrays["face_parts"].dtype),
        "mouth_regions": np.zeros(len(fan), arrays["mouth_regions"].dtype),
    }
    for name in FACE_ARRAYS:
        rows = fan_rows[name].astype(arrays[name].dtype)
        out[name] = np.concatenate([arrays[name][keep], rows])
    if "hair_faces" in arrays:
        if not keep[arrays["hair_faces"]].all():
            raise ValueError("face hair is bound to the canals")
        out["hair_faces"] = renumber[arrays["hair_faces"]].astype(arrays["hair_faces"].dtype)
    if len(arrays.get("gaussian_faces", ())):
        raise ValueError("smooth the crotch of a file without Gaussians")
    return out


def _laplacian(faces: np.ndarray, count: int) -> sparse.csr_matrix:
    """``(V, V)`` the uniform graph Laplacian of the triangles' edges."""
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.concatenate([edges, edges[:, ::-1]])
    adjacency = sparse.coo_matrix((np.ones(len(edges)), edges.T), (count, count)).tocsr()
    adjacency.data[:] = 1
    return sparse.diags(np.asarray(adjacency.sum(axis=1)).ravel()) - adjacency


def _openings(faces: np.ndarray, pinned: np.ndarray) -> list[list[int]]:
    """The loops of boundary edges of ``faces`` that are not pinned: the openings the canals
    left, each as its vertices in order."""
    count: dict[tuple[int, int], int] = defaultdict(int)
    for triangle in faces:
        for i in range(3):
            a, b = triangle[i], triangle[(i + 1) % 3]
            count[(min(a, b), max(a, b))] += 1
    neighbours: dict[int, list[int]] = defaultdict(list)
    for (a, b), n in count.items():
        if n == 1 and not (pinned[a] and pinned[b]):
            neighbours[a].append(b)
            neighbours[b].append(a)

    loops, seen = [], set()
    for start in neighbours:
        if start in seen:
            continue
        loop, previous, current = [start], None, start
        seen.add(start)
        while True:
            following = [v for v in neighbours[current] if v != previous and v not in seen]
            if not following:
                break
            previous, current = current, following[0]
            loop.append(current)
            seen.add(current)
        loops.append([int(v) for v in loop])
    return loops


def _shares_direction(faces: np.ndarray, edge: list[int]) -> bool:
    """Whether a triangle of ``faces`` runs along ``edge`` in the same direction."""
    a, b = edge
    return any(np.any((faces[:, i] == a) & (faces[:, (i + 1) % 3] == b)) for i in range(3))
