"""Finer geometry where the close-ups resolve it: the breast region split into smaller triangles,
and the relief of nipples and areolae fitted to Sapiens2's normals of tight crops around them.

**Refinement** (``refine_region``) is red-green: the chosen triangles split into four at their
edges' midpoints, and so does any triangle with two or three split edges, until none has two;
those with one split into two, so the mesh stays closed. New vertices sit where Loop subdivision
puts an edge's new vertex (3/8 of each end, 1/8 of each opposite corner), and old vertices inside
the refined area move by Loop's rule too (``_smoothed_old``): the refined surface is smooth, not
faceted or rippled; those on its edge stay. Every per-vertex array follows the new vertices by
the same weights (skinning weights kept to the strongest influences, body shapes), per-corner
texture coordinates by interpolation. New vertices go before the head's (``head_start`` grows),
so the body shapes cover them and the head's indices, and every triangle's, keep their meaning: a
split triangle keeps its index for one of its children, the others are appended.

**Relief** (``fit_relief``): within ``ReliefSettings.reach`` of the nipples, the surface moves
along its normals, a little smoothed, to match Sapiens2's normals of crops around each nipple of
the close-ups that show them largest; the broad shape is the shape fit's.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from avatar_splats.records import ArrayRecord

CHILDREN = {
    3: [(0, 3, 5), (3, 1, 4), (5, 4, 2), (3, 4, 5)],
}
"""Corners of a triangle split at all three edges, in the numbering corners 0, 1, 2, then the
midpoints of edges 0-1 (3), 1-2 (4), 2-0 (5)."""


@dataclass(frozen=True)
class Refinement:
    """What one round of ``refine_region`` added, for arrays it does not know."""

    parents: np.ndarray
    """``(n, 4)`` each new vertex's edge ends and opposite corners (-1 where the edge has only one
    triangle), in the old numbering."""
    weights: np.ndarray
    """``(n, 4)`` their weights."""
    start: int
    """Where the new vertices were inserted: the old ``head_start``."""


def _edges_of(faces: np.ndarray) -> np.ndarray:
    """``(F, 3, 2)`` each triangle's edges 0-1, 1-2, 2-0, their ends sorted."""
    pairs = np.stack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=1)
    return np.sort(pairs, axis=-1)


def refine_region(file, chosen: np.ndarray, max_influences: int | None = None):
    """A copy of an ``AvatarFile`` with triangles ``chosen`` (body triangles: all corners before
    ``head_start``) split in four, red-green (see the module's docstring), and the
    ``Refinement`` that says how its new vertices were made.

    Args:
        max_influences: skinning influences kept per vertex; the file's by default.
    """
    from avatar_splats.export.avatar_file import AvatarFile

    arrays, metadata = file.arrays, file.metadata
    head_start = int(metadata["head_start"])
    faces = arrays["faces"].astype(np.int64)
    if (faces[chosen] >= head_start).any():
        raise ValueError("only body triangles can be refined")
    gaussian_faces = arrays.get("gaussian_faces")
    edges = _edges_of(faces)
    keys = edges[..., 0] * len(arrays["rest_vertices"]) + edges[..., 1]
    marked = np.zeros(keys.shape, bool)
    red = np.zeros(len(faces), bool)
    red[chosen] = True
    while True:  # the closure: a triangle with two split edges splits at all three
        split_keys = np.unique(keys[red])
        marked = np.isin(keys, split_keys)
        more = (marked.sum(axis=1) >= 2) & ~red
        if not more.any():
            break
        red |= more
    if (faces[marked.any(axis=1)] >= head_start).any():
        raise ValueError("the refinement reaches the head; choose triangles farther from it")
    if gaussian_faces is not None and np.isin(gaussian_faces, np.flatnonzero(marked.any(1))).any():
        raise ValueError("Gaussians are bound to triangles the refinement would split")

    # the new vertices: one per split edge, Loop's odd rule
    split_keys = np.unique(keys[marked])
    count = len(split_keys)
    index_of = {int(k): head_start + i for i, k in enumerate(split_keys)}
    ends = np.stack([split_keys // len(arrays["rest_vertices"]),
                     split_keys % len(arrays["rest_vertices"])], 1)  # fmt: skip
    opposite = -np.ones((count, 2), np.int64)
    filled = np.zeros(count, np.int64)
    position = np.searchsorted(split_keys, keys)
    for f, k in zip(*np.nonzero(marked), strict=True):
        i = position[f, k]
        if filled[i] < 2:
            opposite[i, filled[i]] = faces[f, (k + 2) % 3]
            filled[i] += 1
    parents = np.concatenate([ends, opposite], axis=1)
    both = (opposite >= 0).all(axis=1)
    weights = np.where(both[:, None], [3 / 8, 3 / 8, 1 / 8, 1 / 8], [0.5, 0.5, 0.0, 0.0])
    weights = weights.astype(np.float64)

    def mixed(values: np.ndarray) -> np.ndarray:
        """``(n, ...)`` the new vertices' values of a per-vertex array (old numbering)."""
        safe = np.maximum(parents, 0)
        return np.einsum("nk,nk...->n...", weights, values[safe]).astype(values.dtype)

    def inserted(values: np.ndarray, new: np.ndarray) -> np.ndarray:
        return np.concatenate([values[:head_start], new, values[head_start:]])

    def renumber(indices: np.ndarray) -> np.ndarray:
        return np.where(indices >= head_start, indices + count, indices)

    # the triangles: each split one keeps its index for its first child
    new_faces = renumber(faces)
    uvs = arrays["uvs"].copy()
    out_faces, out_uvs, out_from = [new_faces], [uvs], [np.arange(len(faces))]
    for f in np.flatnonzero(marked.any(axis=1)):
        corners = list(new_faces[f])
        mids = [index_of.get(int(keys[f, k]), -1) for k in range(3)]
        parent_uv = arrays["uvs"][f]
        corner_uv = list(parent_uv) + [
            (parent_uv[k] + parent_uv[(k + 1) % 3]) / 2 for k in range(3)
        ]
        numbered = corners + mids
        if marked[f].all():
            children = CHILDREN[3]
        else:  # one split edge k: two children
            k = int(np.flatnonzero(marked[f])[0])
            a, b, c = k, (k + 1) % 3, (k + 2) % 3
            children = [(a, 3 + k, c), (3 + k, b, c)]
        rows = np.array([[numbered[i] for i in child] for child in children])
        child_uvs = np.array([[corner_uv[i] for i in child] for child in children])
        new_faces[f] = rows[0]
        uvs[f] = child_uvs[0]
        out_faces.append(rows[1:])
        out_uvs.append(child_uvs[1:])
        out_from.append(np.full(len(rows) - 1, f))
    faces_out = np.concatenate(out_faces).astype(arrays["faces"].dtype)
    origin = np.concatenate(out_from)

    result = dict(arrays)
    result["faces"] = faces_out
    result["uvs"] = np.concatenate(out_uvs).astype(uvs.dtype)
    for name in ("face_textures", "face_parts", "mouth_regions"):
        if name in arrays:
            result[name] = arrays[name][origin]
    rest = _smoothed_old(arrays["rest_vertices"], faces, red, head_start)
    result["rest_vertices"] = inserted(rest, mixed(arrays["rest_vertices"]))
    indices, bone_weights = _mixed_skinning(
        arrays["bone_indices"], arrays["bone_weights"], parents, weights,
        max_influences or arrays["bone_indices"].shape[1],
    )  # fmt: skip
    result["bone_indices"] = inserted(arrays["bone_indices"], indices)
    result["bone_weights"] = inserted(arrays["bone_weights"], bone_weights)
    if "body_shape_basis" in arrays:
        basis = arrays["body_shape_basis"]
        result["body_shape_basis"] = np.concatenate([basis, np.stack([mixed(b) for b in basis])], 1)
    metadata = dict(metadata, head_start=head_start + count)
    if "eyes" in metadata:
        eyes = dict(metadata["eyes"])
        eyes["eyeballs"] = [
            dict(e, apex=int(e["apex"]) + count, top=int(e["top"]) + count)
            for e in eyes["eyeballs"]
        ]
        metadata["eyes"] = eyes
    return AvatarFile(metadata, result), Refinement(parents, weights, head_start)


def _smoothed_old(rest: np.ndarray, faces: np.ndarray, red: np.ndarray, head_start: int):
    """``rest`` with Loop's rule for old vertices on the body vertices whose triangles all split in
    four: ``(1 - n b) v + b`` times the sum of its ``n`` neighbours, ``b = 3 / (8 n)`` (``3 / 16``
    for three). New vertices smoothed and old ones not would alternate, and a second round would
    turn that into ripples. Vertices on the refined area's edge stay, so it meets the rest."""
    out = rest.copy()
    touched = np.zeros(len(rest), bool)
    touched[faces[~red].ravel()] = True
    inside = np.zeros(len(rest), bool)
    inside[faces[red].ravel()] = True
    moving = np.flatnonzero(inside & ~touched & (np.arange(len(rest)) < head_start))
    if not len(moving):
        return out
    edges = np.concatenate([faces[red][:, [0, 1]], faces[red][:, [1, 2]], faces[red][:, [2, 0]]])
    edges = np.unique(np.sort(np.concatenate([edges, edges[:, ::-1]]), axis=1), axis=0)
    edges = np.concatenate([edges, edges[:, ::-1]])
    sums = np.zeros_like(rest, dtype=np.float64)
    np.add.at(sums, edges[:, 0], rest[edges[:, 1]])
    count = np.bincount(edges[:, 0], minlength=len(rest)).astype(np.float64)
    n = count[moving]
    beta = np.where(n == 3, 3 / 16, 3 / (8 * np.maximum(n, 1)))
    out[moving] = ((1 - n * beta)[:, None] * rest[moving] + beta[:, None] * sums[moving]).astype(
        rest.dtype
    )
    return out


def _mixed_skinning(indices, weights, parents, mix, keep: int):
    """``(n, keep)`` bone indices and weights of new vertices: their parents' weights mixed,
    the ``keep`` strongest renormalized."""
    count = len(parents)
    out_indices = np.zeros((count, keep), indices.dtype)
    out_weights = np.zeros((count, keep), weights.dtype)
    for i in range(count):
        total: dict[int, float] = {}
        for parent, share in zip(parents[i], mix[i], strict=True):
            if parent < 0 or share == 0:
                continue
            for bone, w in zip(indices[parent], weights[parent], strict=True):
                if w > 0:
                    total[int(bone)] = total.get(int(bone), 0.0) + share * float(w)
        strongest = sorted(total.items(), key=lambda kv: -kv[1])[:keep]
        norm = sum(w for _, w in strongest) or 1.0
        for k, (bone, w) in enumerate(strongest):
            out_indices[i, k] = bone
            out_weights[i, k] = w / norm
    return out_indices, out_weights


def extended(values: np.ndarray, refinement: Refinement) -> np.ndarray:
    """A per-vertex array of the body (``(B, ...)``, before ``head_start``) extended to the
    refined body: its new vertices' values mixed as ``refine_region`` mixed their positions."""
    safe = np.maximum(refinement.parents, 0)
    new = np.einsum("nk,nk...->n...", refinement.weights, values[safe]).astype(values.dtype)
    return np.concatenate([values, new])


@dataclass(frozen=True)
class ReliefSettings:
    count: int
    """Close-ups the relief is fitted to: those that show the nipples largest."""
    crop_span: float
    """A nipple's crop's side over the nipples' spread."""
    crop_size: int
    """Pixels along a nipple's crop's side (Sapiens2's normals at the same)."""
    render_size: int
    steps: int
    learning_rate: float
    smoothing_steps: int
    """Rounds of averaging the relief's free values: few, it is fine detail."""
    reach: float
    """Metres from a nipple within which the surface may move, fading out towards it."""
    min_identity: float
    weights: dict[str, float]
    """Per loss its weight: normals, outline, nipples, pose, relief (per square millimetre)."""


@dataclass(frozen=True)
class Relief(ArrayRecord):
    change: np.ndarray
    """``(B, 3)`` metres added to the refined body's vertices: along their normals."""
    keys: np.ndarray
    """``(M,)`` the close-ups the nipples' crops came from (each nipple its own crop)."""
    normal_before: np.ndarray
    normal_after: np.ndarray
    """``(M,)`` mean degrees between rendered and Sapiens2's normals over the breasts, the crops
    placed again, without and with the relief."""


def nipple_crops(
    closeups, places, settings: ReliefSettings, video_paths: dict, parts_model: str,
    normal_model: str, device: str,
):  # fmt: skip
    """``Closeups`` of tight crops around each nipple of the ``settings.count`` close-ups (both
    nipples found, likeness known) that show them largest, and ``(M,)`` the index of the
    close-up each came from. ``places``: the fit whose close-ups are placed (its ``iou_after``
    must reach 0.6)."""
    import torch

    from avatar_splats.data.video import read_frames
    from avatar_splats.preprocess.prepare import BodyParts
    from avatar_splats.regions.closeups import (
        Closeups,
        crop_intrinsics,
        cut,
        estimate_normals,
        frame_intrinsics,
    )

    s = settings
    scale = (closeups.crops[:, 2] - closeups.crops[:, 0]) / closeups.images.shape[1]
    spread = np.linalg.norm(closeups.nipples[:, 0] - closeups.nipples[:, 1], axis=1) * scale
    usable = np.isfinite(spread) & (np.nan_to_num(closeups.identity, nan=-1) >= s.min_identity)
    usable &= places.iou_after >= 0.6
    chosen = np.argsort(-np.where(usable, spread, -1))[: min(s.count, usable.sum())]
    rows = []
    for i in chosen:
        frame = next(read_frames(video_paths[str(closeups.videos[i])], max_frames=1,
                                 start=int(closeups.frames[i])))  # fmt: skip
        for nipple in closeups.nipples[i]:
            centre = closeups.crops[i, :2] + nipple * scale[i]
            half = s.crop_span * spread[i] / 2
            crop = np.array([*(centre - half), *(centre + half)])
            K = crop_intrinsics(frame_intrinsics(closeups, i), crop, s.crop_size)
            rows.append((i, cut(frame, crop, s.crop_size), crop, K))
    images = np.stack([r[1] for r in rows])
    whole = np.array([[0.0, 0.0, s.crop_size, s.crop_size]] * len(images))
    labels, _ = BodyParts(parts_model, 8, 0.0, device)(list(images), whole)
    torch.cuda.empty_cache()
    count = len(rows)
    nipples = np.full((count, 2, 2), np.nan, np.float32)
    nipples[:, 0] = s.crop_size / 2  # each crop centred on its nipple
    crops = Closeups(
        keys=closeups.keys[[r[0] for r in rows]], videos=closeups.videos[[r[0] for r in rows]],
        frames=closeups.frames[[r[0] for r in rows]], images=images,
        K=np.stack([r[3] for r in rows]).astype(np.float32),
        crops=np.stack([r[2] for r in rows]).astype(np.float32),
        labels=np.asarray(labels, np.uint8),
        normals=estimate_normals(images, normal_model, device),
        breasts=np.zeros(images.shape[:3], bool), areolae=np.zeros(images.shape[:3], bool),
        nipples=nipples, keypoints=np.zeros((count, 17, 3), np.float32),
        identity=closeups.identity[[r[0] for r in rows]],
    )  # fmt: skip
    return crops, np.array([r[0] for r in rows])


def fit_relief(
    body, file, bone_labels: list[str], weights: np.ndarray, nipples: np.ndarray, crops,
    sources: np.ndarray, places, settings: ReliefSettings, device: str, progress: bool = True,
) -> Relief:  # fmt: skip
    """Fit the relief of the refined breast region to the nipples' crops.

    Args:
        body: an ``AnnyBody`` on ``device``.
        file: the refined ``AvatarFile`` (its ``body_shape_basis`` the posture correctives).
        weights: ``(B,)`` the breast region's weight on the refined body.
        nipples: ``(2,)`` the nipples' body vertices.
        crops, sources: ``nipple_crops``'.
        places: the ``GravityFit`` that placed the close-ups and weighted their correctives.
    """
    import torch
    from tqdm.auto import tqdm

    from avatar_splats.body_models.mesh import vertex_normals
    from avatar_splats.regions.breasts import MILLIMETRE, Scene, averaging_matrix, smooth

    s = settings
    arrays = dict(file.arrays, head_start=file.metadata["head_start"])
    weight_t = torch.from_numpy(weights).float().to(device)
    scene = Scene(
        body, arrays, bone_labels, crops, weight_t, torch.from_numpy(nipples).long().to(device),
        s.render_size, device,
    )  # fmt: skip
    rest = scene.rest
    basis = torch.from_numpy(file.arrays["body_shape_basis"]).float().to(device)
    shapes = torch.einsum(
        "ns,sbc->nbc", torch.from_numpy(places.weights[sources]).float().to(device), basis
    )
    placed = [torch.nn.Parameter(torch.from_numpy(getattr(places, k)[sources]).float().to(device))
              for k in ("rotations", "translations", "turns")]  # fmt: skip
    normals = vertex_normals(rest[None], scene.faces)[0]
    distance = torch.cdist(rest, rest[torch.from_numpy(nipples).long().to(device)]).min(1).values
    free = (1 - distance / s.reach).clamp(0, 1).sqrt()[:, None]
    matrix = averaging_matrix(scene.faces, len(rest))
    values = torch.nn.Parameter(torch.zeros(len(rest), 1, device=device))

    def relief() -> torch.Tensor:
        """``(B, 1)`` metres along the normals."""
        return smooth(values * free, matrix, s.smoothing_steps) * free * MILLIMETRE

    def losses(scores: bool = False) -> dict:
        vertices, transforms = scene.posing(rest + relief() * normals + shapes, *placed)
        return scene.losses(vertices, transforms, scores)

    w = s.weights

    def run(parameters: list, desc: str) -> None:
        # the crops placed again first (at the nipples' scale the close-ups' places are a few
        # pixels off, which the relief would take up), then placed and relieved together
        optimiser = torch.optim.Adam(parameters, lr=s.learning_rate)
        for _ in tqdm(range(s.steps), desc=desc, disable=not progress, leave=False):
            optimiser.zero_grad()
            parts = losses()
            loss = sum(w[k] * parts[k].mean() for k in ("normals", "outline", "nipples"))
            loss = loss + w["pose"] * placed[2].square().sum((1, 2)).mean()
            loss = loss + w["relief"] * relief().div(MILLIMETRE).square().mean()
            loss.backward()
            optimiser.step()

    placing = [{"params": placed, "lr": s.learning_rate / 10}]
    run(placing, "placing")
    with torch.no_grad():
        before = losses(scores=True)
    run([*placing, {"params": [values]}], "relief")
    with torch.no_grad():
        after = losses(scores=True)
        change = (relief() * normals).cpu().numpy()
    return Relief(
        change=change,
        keys=crops.keys,
        normal_before=before["angle"].cpu().numpy(),
        normal_after=after["angle"].cpu().numpy(),
    )


def with_relief(file, relief: Relief):
    """A copy of an ``AvatarFile`` with ``relief``'s change on its body vertices."""
    from avatar_splats.export.avatar_file import AvatarFile

    head_start = int(file.metadata["head_start"])
    rest = file.arrays["rest_vertices"].copy()
    rest[:head_start] += relief.change.astype(rest.dtype)
    return AvatarFile(dict(file.metadata), dict(file.arrays, rest_vertices=rest))


def refine_breasts(file, weights: np.ndarray, nipples: np.ndarray, reach: float):
    """``refine_region`` twice: the breast region (``texture.region_faces``), then within
    ``reach`` metres of the nipples. Returns the refined file and ``weights`` (the breast
    region's, ``(B,)`` on the body) extended to its body."""
    from avatar_splats.regions.texture import region_faces

    head_start = int(file.metadata["head_start"])
    file, first = refine_region(
        file, region_faces(file.arrays["faces"].astype(np.int64), weights, head_start)
    )
    weights = extended(weights, first)
    head_start = int(file.metadata["head_start"])
    faces = file.arrays["faces"].astype(np.int64)
    rest = file.arrays["rest_vertices"][:head_start]
    near = np.linalg.norm(rest[:, None] - rest[nipples][None], axis=-1).min(axis=1) < reach
    body = (faces < head_start).all(axis=1)
    inner = np.flatnonzero(body & near[np.minimum(faces, head_start - 1)].all(axis=1))
    file, second = refine_region(file, inner)
    return file, extended(weights, second)
