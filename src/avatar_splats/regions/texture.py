"""The breasts' own texture, baked from the close-ups at many times the body texture's detail and
blended into it without a seam.

**The region** is the body's triangles with a corner in the breast region (``BreastRegion``'s
weight above ``REGION_WEIGHT``): one disc over both breasts and the chest between them. It is laid
flat by a harmonic map (``harmonic_uvs``: its boundary on a circle, cotangent weights) into its
own texture, ``breast_texture``, a third one beside the body's and the head's.

**Albedo.** Each close-up's light is fitted (second-order spherical harmonics, as notebook 12's,
``fit_light``) to its colours, the surface's normals and the body texture's albedo there
(``base``), and divided out: what is left is the skin's own colour, without the sun's highlights
and the shadow under the breasts that every close-up of a sunlit video shares. The albedo's
logarithm is averaged over the close-ups that see a texel, weighted by how squarely and how
closely they see it, robustly (``robust_mean``): a strand of hair over the skin in a few
close-ups drops out.

**Blending.** Changes of colour broader than ``match_mm`` are the body texture's (its albedo and
the close-ups' differ by the cameras' white balance and exposure, and by what the light fits
left), finer ones the close-ups'; towards the region's edge the texture fades into the body
texture's (the region's weight): no seam.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from avatar_splats.records import ArrayRecord

REGION_WEIGHT = 0.02
"""A triangle takes the breast texture where a corner's breast region weight is above this."""
TEXTURE_NAME = "breast_texture"
MIN_SHADING = 0.35
"""The least light, as a share of a close-up's median over the region, its colours are divided
by."""
MARGIN = 0.02
"""Share of the texture's side left around the region's disc."""


def region_faces(faces: np.ndarray, weights: np.ndarray, head_start: int) -> np.ndarray:
    """``(R,)`` the indices of the avatar's triangles in the breast region (``REGION_WEIGHT``)."""
    body = (faces < head_start).all(axis=1)
    corner = weights[np.minimum(faces, head_start - 1)].max(axis=1)
    return np.flatnonzero(body & (corner > REGION_WEIGHT))


def harmonic_uvs(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """``(R, 3, 2)`` texture coordinates in [0, 1] for a disc of ``(R, 3)`` triangles of ``(V, 3)``
    vertices: its boundary on a circle by arc length, the inside where cotangent weights (clamped
    to stay positive, so no triangle flips) put it, then turned and mirrored so that u follows x
    and v follows z as closely as a rotation allows."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve

    from avatar_splats.body_models.mesh import boundary_loops

    used, local = np.unique(faces, return_inverse=True)
    local = local.reshape(faces.shape)
    points = vertices[used].astype(np.float64)
    loop = max(boundary_loops(local), key=len)
    steps = np.linalg.norm(points[np.roll(loop, -1)] - points[loop], axis=1)
    angle = 2 * np.pi * np.concatenate([[0.0], np.cumsum(steps)[:-1]]) / steps.sum()
    uv = np.zeros((len(used), 2))
    uv[loop] = np.stack([np.cos(angle), np.sin(angle)], axis=1)
    rows, columns, values = [], [], []
    for k in range(3):
        i, j, o = local[:, k], local[:, (k + 1) % 3], local[:, (k + 2) % 3]
        a, b = points[i] - points[o], points[j] - points[o]
        cot = (a * b).sum(1) / np.maximum(np.linalg.norm(np.cross(a, b), axis=1), 1e-12)
        weight = np.maximum(cot / 2, 1e-6)
        rows += [i, j]
        columns += [j, i]
        values += [weight, weight]
    rows, columns, values = (np.concatenate(x) for x in (rows, columns, values))
    count = len(used)
    weights = coo_matrix((values, (rows, columns)), shape=(count, count)).tocsr()
    laplacian = (
        coo_matrix(
            (np.asarray(weights.sum(axis=1)).ravel(), (np.arange(count), np.arange(count))),
            shape=(count, count),
        ).tocsr()
        - weights
    )
    inner = np.setdiff1d(np.arange(count), loop)
    fixed = laplacian[inner][:, loop] @ uv[loop]
    system = laplacian[inner][:, inner].tocsc()
    uv[inner] = np.stack([spsolve(system, -fixed[:, c]) for c in range(2)], axis=1)
    # turn (and mirror) so u follows x and v follows z: the texture reads as she is seen
    target = points[:, [0, 2]] - points[:, [0, 2]].mean(0)
    u, _, vt = np.linalg.svd(uv.T @ target)
    uv = uv @ (u @ vt)
    uv = (uv - uv.min(0)) / (uv.max(0) - uv.min(0)).max()
    uv = MARGIN + (1 - 2 * MARGIN) * uv
    return uv[local].astype(np.float32)


@dataclass(frozen=True)
class BakeSettings:
    size: int
    """Texels along the texture's side."""
    crop_size: int
    """Pixels along a close-up's side as baked from (at most the source's)."""
    match_mm: float
    """Millimetres: changes of colour broader than this are the body texture's, finer ones the
    close-ups'."""
    min_cosine: float
    """Least cosine between a texel's normal and the view for a close-up to count there."""
    depth_tolerance: float
    """Metres a texel may lie behind the drawn surface and still be seen."""
    robust_steps: int
    """Rounds of reweighting the close-ups' detail per texel (``robust_mean``)."""
    robust_scale: float
    """How far, in the detail's logarithm, a close-up's value may lie from the others' before
    it counts less: a strand of hair over the skin in a few close-ups drops out."""
    min_identity: float
    min_iou: float
    """As the shape fit's: which close-ups are baked from."""


@dataclass(frozen=True)
class BreastTexture(ArrayRecord):
    faces: np.ndarray
    """``(R,)`` the avatar's triangles that take the breast texture."""
    uvs: np.ndarray
    """``(R, 3, 2)`` their corners' coordinates in it (v up)."""
    texture: np.ndarray
    """``(S, S, 3)`` float in [0, 1], row 0 at v = 0."""
    base: np.ndarray
    """``(S, S, 3)`` the body texture's colour at the same points: the texture without detail."""
    coverage: np.ndarray
    """``(S, S)`` how much the close-ups saw each texel (summed weights); 0 unseen."""
    inside: np.ndarray
    """``(S, S)`` bool: the texels on the region's triangles."""
    keys: np.ndarray
    """``(N,)`` the close-ups baked from."""


def _rasterize_uv(uvs: torch.Tensor, size: int):
    """nvdiffrast's rasterization of ``(R, 3, 2)`` corner coordinates into a ``size`` square:
    ``(size, size)`` triangle (-1: none) and ``(size, size, 2)`` its barycentrics (``_surface``);
    row 0 at v = 0."""
    import nvdiffrast.torch as dr

    corners = uvs.reshape(-1, 2) * 2 - 1
    clip = torch.cat(
        [corners, torch.zeros_like(corners[:, :1]), torch.ones_like(corners[:, :1])], 1
    )
    tri = torch.arange(len(corners), device=uvs.device, dtype=torch.int32).reshape(-1, 3)
    rast, _ = dr.rasterize(dr.RasterizeCudaContext(), clip[None].contiguous(), tri, (size, size))
    face = rast[0, ..., 3].long() - 1
    return face, rast[0, ..., :2]


def _surface(values: torch.Tensor, face: torch.Tensor, bary: torch.Tensor) -> torch.Tensor:
    """``(..., C)`` per texel the mix of ``(R, 3, C)`` corner values by nvdiffrast's barycentrics
    (``(u, v)``: the weights of corners 0 and 1; corner 2 has the rest)."""
    corners = values[face.clamp_min(0)]
    u, v = bary[..., 0:1], bary[..., 1:2]
    return u * corners[..., 0, :] + v * corners[..., 1, :] + (1 - u - v) * corners[..., 2, :]


def _sample(image: torch.Tensor, pixels: torch.Tensor) -> torch.Tensor:
    """``(P, C)`` bilinear samples of a ``(C, H, W)`` image at ``(P, 2)`` pixel positions."""
    height, width = image.shape[1:]
    grid = torch.stack(
        [(pixels[:, 0] + 0.5) / width * 2 - 1, (pixels[:, 1] + 0.5) / height * 2 - 1], -1
    )
    return F.grid_sample(image[None], grid[None, None], align_corners=False)[0, :, 0].T


def push_pull(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """``(H, W, C)`` weighted ``values`` with the texels of zero weight filled from their
    surroundings (averaged down a pyramid, then filled from the coarsest level that has one)."""
    levels = [(values.permute(2, 0, 1)[None] * weights[None, None], weights[None, None])]
    while min(levels[-1][1].shape[-2:]) > 1:
        sums, counts = levels[-1]
        levels.append(
            (F.avg_pool2d(sums, 2, ceil_mode=True), F.avg_pool2d(counts, 2, ceil_mode=True))
        )
    sums, counts = levels[-1]
    filled = sums / counts.clamp_min(1e-12)
    for sums, counts in reversed(levels[:-1]):
        up = F.interpolate(filled, sums.shape[-2:], mode="bilinear", align_corners=False)
        own = sums / counts.clamp_min(1e-12)
        known = (counts > 1e-8).float()
        filled = known * own + (1 - known) * up
    return filled[0].permute(1, 2, 0)


@torch.no_grad()
def bake_breasts(
    body, arrays: dict, bone_labels: list[str], closeups, fit, gravity,
    video_paths: dict[str, Path], settings: BakeSettings, device: str, progress: bool = True,
) -> BreastTexture:  # fmt: skip
    """Bake the breast texture from ``closeups`` placed as ``gravity`` (or, None, ``fit``) left
    them, each with its posture correctives.

    Args:
        body: an ``AnnyBody`` on ``device``.
        arrays: the ``.avatar`` file's arrays (with ``body_texture``) and ``head_start``.
        fit: the ``BreastFit``; gravity: the ``GravityFit`` or None.
        video_paths: per harvest video name its file.
    """
    from avatar_splats.body_models.mesh import vertex_normals
    from avatar_splats.data.video import read_frames
    from avatar_splats.preprocess.prepare import PART_NAMES
    from avatar_splats.regions.breasts import Scene
    from avatar_splats.regions.closeups import crop_intrinsics, cut, frame_intrinsics

    s = settings
    head_start = int(arrays["head_start"])
    faces_np = arrays["faces"].astype(np.int64)
    chosen = region_faces(faces_np, fit.weights, head_start)
    rest_np = arrays["rest_vertices"][:head_start] + fit.change
    uvs = harmonic_uvs(rest_np, faces_np[chosen])
    face, bary = _rasterize_uv(torch.from_numpy(uvs).to(device), s.size)
    inside = face >= 0
    faces = torch.from_numpy(faces_np[chosen]).to(device)

    # the base: the body texture where these texels are on its own layout
    body_uvs = torch.from_numpy(arrays["uvs"][chosen]).float().to(device)
    texel_uv = _surface(body_uvs, face, bary)
    body_texture = (
        torch.from_numpy(arrays["body_texture"]).float().to(device).permute(2, 0, 1) / 255
    )
    height, width = body_texture.shape[1:]
    pixels = torch.stack(
        [texel_uv[..., 0] * width - 0.5, (1 - texel_uv[..., 1]) * height - 0.5], -1
    )
    base = _sample(body_texture, pixels.reshape(-1, 2)).reshape(s.size, s.size, 3)

    weights = torch.from_numpy(fit.weights).float().to(device)
    nipples = torch.from_numpy(fit.nipples).long().to(device)
    scene = Scene(body, arrays, bone_labels, closeups, weights, nipples, 256, device)
    rest = scene.rest + torch.from_numpy(fit.change).float().to(device)
    places = gravity if gravity is not None else fit
    shapes = (
        torch.einsum(
            "ns,sbc->nbc", torch.from_numpy(gravity.weights).float().to(device),
            torch.from_numpy(gravity.shapes).float().to(device),
        )
        if gravity is not None else torch.zeros(len(fit.keys), *rest.shape, device=device)
    )  # fmt: skip
    fade = _surface(weights[faces][..., None], face, bary)[..., 0].clamp(0, 1)
    identity = np.nan_to_num(closeups.identity, nan=-1.0)
    iou = places.iou_after
    used = np.flatnonzero((identity >= s.min_identity) & (iou >= s.min_iou))
    skin_label = PART_NAMES.index("Torso")
    samples = []  # per close-up: the texels it sees, its detail there and its weight
    for i in tqdm(used, desc="baking", unit="close-up", disable=not progress):
        crop = closeups.crops[i]
        frame = next(read_frames(video_paths[str(closeups.videos[i])], max_frames=1,
                                 start=int(closeups.frames[i])))  # fmt: skip
        side = int(min(s.crop_size, round(crop[2] - crop[0])))
        image = torch.from_numpy(cut(frame, crop, side)).float().to(device).permute(2, 0, 1) / 255
        K = crop_intrinsics(frame_intrinsics(closeups, i), crop, side)
        K = torch.from_numpy(K).float().to(device)
        vertices, _ = scene.posing(
            rest + shapes[i], *(torch.from_numpy(getattr(places, k)[i : i + 1]).float().to(device)
                                for k in ("rotations", "translations", "turns")),
        )  # fmt: skip
        vertices = vertices[0]
        points = _surface(vertices[faces], face, bary)
        normals = F.normalize(
            _surface(vertex_normals(vertices[None], scene.faces)[0][faces], face, bary), dim=-1
        )
        uv = points @ K.T
        texel_pixels = uv[..., :2] / uv[..., 2:].clamp_min(1e-6)
        depth = _depth(scene, vertices, K, side)
        px = texel_pixels.round().long()
        on = (
            inside
            & (px[..., 0] >= 0)
            & (px[..., 0] < side)
            & (px[..., 1] >= 0)
            & (px[..., 1] < side)
        )
        seen_depth = depth[px[..., 1].clamp(0, side - 1), px[..., 0].clamp(0, side - 1)]
        on &= points[..., 2] <= seen_depth + s.depth_tolerance
        cosine = -(normals * F.normalize(points, dim=-1)).sum(-1)
        on &= cosine >= s.min_cosine
        labels = torch.from_numpy(closeups.labels[i]).to(device)
        scale = labels.shape[0] / side
        lx = (texel_pixels[..., 0] * scale).long().clamp(0, labels.shape[1] - 1)
        ly = (texel_pixels[..., 1] * scale).long().clamp(0, labels.shape[0] - 1)
        on &= labels[ly, lx] == skin_label
        if not on.any():
            continue
        per_mm = K[0, 0] / points[..., 2].clamp_min(1e-3) / 1000
        colour = _sample(image, texel_pixels[on])
        shading = sh_basis(normals[on]) @ fit_light(colour, base[on], normals[on])
        # where the light leaves little, bounce light and the breast's own shadow (neither in
        # the light's model) decide the colour: kept from blowing up, those samples count less
        brightness = shading.mean(-1, keepdim=True)
        floor = MIN_SHADING * brightness.median()
        albedo = torch.log(colour / shading.clamp_min(floor) + 0.01)
        lit = (brightness[:, 0] / brightness.median()).clamp(0, 1)
        weight = cosine[on] ** 4 * per_mm[on].square() * lit.square()
        texels = torch.nonzero(on.flatten())[:, 0]
        samples.append((texels.cpu(), albedo.half().cpu(), weight.cpu()))
    mean, total = robust_mean(samples, s.size * s.size, s.robust_steps, s.robust_scale, device)
    mean, total = mean.reshape(s.size, s.size, 3), total.reshape(s.size, s.size)
    seen = push_pull(mean, (total > 0).float())
    # the close-ups' albedo, its broadest changes the body texture's, faded into it at the edge
    log_base = torch.log(base + 0.01)
    area = _area(rest_np, faces_np[chosen])
    sigma = s.match_mm / (1000 * math.sqrt(area / max(int(inside.sum()), 1)))
    matched = seen + _masked_blur(log_base - seen, inside, sigma)
    blended = fade[..., None] * matched + (1 - fade[..., None]) * log_base
    texture = torch.where(inside[..., None], torch.exp(blended) - 0.01, base).clamp(0, 1)
    return BreastTexture(
        faces=chosen, uvs=uvs, texture=texture.cpu().numpy(), base=base.cpu().numpy(),
        coverage=total.cpu().numpy(), inside=inside.cpu().numpy(), keys=closeups.keys[used],
    )  # fmt: skip


def robust_mean(samples, count: int, steps: int, scale: float, device: str):
    """``(count, C)`` per texel a Huber-weighted mean of the samples (per close-up: ``(P,)``
    texels, ``(P, C)`` values, ``(P,)`` weights), reweighted ``steps`` times: a value farther than
    ``scale`` from the texel's mean counts in inverse proportion to its distance. And ``(count,)``
    the summed weights."""
    channels = samples[0][1].shape[1] if samples else 3
    mean = torch.zeros(count, channels, device=device)
    for step in range(steps + 1):
        sums = torch.zeros(count, channels, device=device)
        total = torch.zeros(count, device=device)
        for texels, values, weights in samples:
            texels, values = texels.to(device), values.to(device).float()
            weights = weights.to(device)
            if step:
                distance = (values - mean[texels]).norm(dim=-1)
                weights = weights * (scale / distance.clamp_min(scale))
            sums.index_add_(0, texels, weights[:, None] * values)
            total.index_add_(0, texels, weights)
        mean = sums / total.clamp_min(1e-12)[:, None]
    return mean, total


def sh_basis(normals: torch.Tensor) -> torch.Tensor:
    """``(..., 9)`` real spherical harmonics up to degree 2 of ``(..., 3)`` unit normals."""
    x, y, z = normals.unbind(-1)
    return torch.stack(
        [0.282095 * torch.ones_like(x), 0.488603 * y, 0.488603 * z, 0.488603 * x,
         1.092548 * x * y, 1.092548 * y * z, 0.315392 * (3 * z * z - 1), 1.092548 * x * z,
         0.546274 * (x * x - y * y)],
        -1,
    )  # fmt: skip


def fit_light(colour: torch.Tensor, albedo: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
    """``(9, 3)`` the spherical-harmonic light that best turns ``(P, 3)`` albedo on surfaces
    of ``(P, 3)`` normals into ``(P, 3)`` colours (least squares per channel, a little ridge)."""
    basis = sh_basis(normals)
    out = []
    for c in range(3):
        design = basis * albedo[:, c : c + 1]
        gram = design.T @ design
        gram = gram + 1e-4 * gram.diagonal().mean() * torch.eye(9, device=design.device)
        out.append(torch.linalg.solve(gram, design.T @ colour[:, c]))
    return torch.stack(out, 1)


def _area(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Square metres of ``(R, 3)`` triangles."""
    corners = vertices[faces]
    cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    return float(np.linalg.norm(cross, axis=1).sum() / 2)


def _masked_blur(values: torch.Tensor, mask: torch.Tensor, sigma: float) -> torch.Tensor:
    """``(H, W, C)`` a Gaussian blur of ``sigma`` texels averaging only texels in the mask."""
    weights = mask.float()[None]
    sums = _blur(values.permute(2, 0, 1) * weights, sigma)
    return (sums / _blur(weights, sigma).clamp_min(1e-6)).permute(1, 2, 0)


def _blur(image: torch.Tensor, sigma: float) -> torch.Tensor:
    """``(C, H, W)`` a Gaussian blur of ``sigma`` pixels."""
    radius = max(1, math.ceil(3 * sigma))
    offsets = torch.arange(-radius, radius + 1, device=image.device, dtype=image.dtype)
    kernel = torch.exp(-0.5 * (offsets / max(sigma, 1e-3)).square())
    kernel = kernel / kernel.sum()
    channels = image[:, None]
    channels = F.conv2d(
        F.pad(channels, (0, 0, radius, radius), mode="replicate"), kernel.view(1, 1, -1, 1)
    )
    channels = F.conv2d(
        F.pad(channels, (radius, radius, 0, 0), mode="replicate"), kernel.view(1, 1, 1, -1)
    )
    return channels[:, 0]


def _depth(scene, vertices: torch.Tensor, K: torch.Tensor, size: int) -> torch.Tensor:
    """``(size, size)`` the drawn torso's depth along the view, infinite where none."""
    import nvdiffrast.torch as dr

    from avatar_splats.regions.breasts import clip

    positions = clip(vertices[None], K[None], size)
    faces = scene.renderer.faces
    rast, _ = dr.rasterize(scene.renderer.context, positions, faces, (size, size))
    depth, _ = dr.interpolate(vertices[None, :, 2:].contiguous(), rast, faces)
    return torch.where(
        rast[0, ..., 3] > 0, depth[0, ..., 0], torch.full_like(depth[0, ..., 0], math.inf)
    )


def with_breast_texture(file, texture: BreastTexture):
    """A copy of an ``AvatarFile`` whose breast region takes ``texture``: a third texture,
    ``breast_texture`` (8-bit, row 0 at the top, v = 1), its triangles' ``face_textures`` and
    ``uvs`` pointing at it."""
    from avatar_splats.export.avatar_file import AvatarFile

    metadata = dict(file.metadata)
    names = list(metadata["textures"])
    if TEXTURE_NAME not in names:
        names.append(TEXTURE_NAME)
    metadata["textures"] = names
    face_textures = file.arrays["face_textures"].copy()
    face_textures[texture.faces] = names.index(TEXTURE_NAME)
    uvs = file.arrays["uvs"].copy()
    uvs[texture.faces] = texture.uvs
    image = np.round(np.clip(texture.texture[::-1], 0, 1) * 255).astype(np.uint8)
    arrays = dict(file.arrays, face_textures=face_textures, uvs=uvs)
    arrays[TEXTURE_NAME] = np.ascontiguousarray(image)
    return AvatarFile(metadata, arrays)
