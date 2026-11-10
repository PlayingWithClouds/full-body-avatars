"""Eyelashes and eyebrows as strands, bound to the face so they follow blinks and expressions.

Neither is captured: lashes are too fine for the videos, and brows baked into the texture look
painted on. Both are grown here and drawn by the runtime as thin strands
(the viewer's is ``viewer/src/lib/render/face_hair.ts``).

- Lashes follow ICT-FaceKit's eyelash cards, three layered strips along each lid, each a grid
  of columns running from a lash's root to its tip: the lashes are its columns, interpolated
  along the lid, with some spread in spacing, length and direction.
- Brows grow where the performer has them: in a band along the model's brow landmarks, where
  the baked head texture is darker than the skin around it. They point up and out at the inner
  end, out along the rest, the upper and lower ones converging on the brow's middle line, and
  lie almost flat on the skin.

Each strand is bound to the skin triangle nearest its root: its points are stored in that
triangle's frame (first edge, across, normal; see ``triangle_frames``), in metres from the root,
and the runtime places them on the posed triangle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sparse
import torch
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.spatial import cKDTree

from avatar_splats.body_models import ict
from avatar_splats.body_models.hybrid import EYES, HEAD_SKIN, HEAD_TEXTURE, Avatar, _umeyama
from avatar_splats.body_models.mesh import closest_points, triangle_frames
from avatar_splats.export.eyes import frontal
from avatar_splats.training.textures import TextureRecord

POINTS = 6
"""Points per strand, root first."""
LASHES_PER_CARD = {"upper": 40, "lower": 7}
LASH_LENGTH = (0.75, 1.05)
"""Each lash's length, in shares of its card's, drawn uniformly."""
LASH_SPREAD = 0.08
"""Standard deviation, in radians, of how far a lash turns from its card's direction."""
LASH_WIDTH = 9e-5
"""Metres across at the root; strands taper to a point."""
ROOT_TOLERANCE = 2e-4
"""How close to the skin, in metres, a card's vertex is to count as a root."""

BROW_HAIRS = 450
"""Hairs per brow, at most."""
BROW_BAND = (-0.010, -0.001)
"""Where brow hairs may grow, in metres from the line of the model's brow landmarks (negative
below it): the line runs along a brow's upper edge, and hair rooted over plain skin shows as
stray wisps."""
BROW_CANDIDATES = 20000
"""Points tried per brow."""
BROW_CONTRAST = 0.25
"""A texel is fully brow where it is this much darker than the skin around, in shares of the
skin's brightness."""
BROW_MIN_DARKNESS = 0.35
"""Hair grows only where the texture is at least this much brow."""
BROW_SHADE = 0.8
"""Brow hairs are drawn this much darker than the brow measured in the video: over the baked
brow, which has that colour, hairs of the same colour would not show."""
BROW_LENGTH = (0.005, 0.0075, 0.0055)
"""Metres long at the inner end, the middle and the outer end of a brow."""
BROW_LIFT = (0.5, 0.22)
"""Radians out of the skin at the inner end and elsewhere."""
BROW_WIDTH = 6e-5
BROW_LANDMARKS = ((17, 18, 19, 20, 21), (22, 23, 24, 25, 26))
"""The Multi-PIE landmarks along each brow."""

MEDIAPIPE_BROWS = (
    (70, 63, 105, 66, 107, 55, 65, 52, 53, 46),
    (300, 293, 334, 296, 336, 285, 295, 282, 283, 276),
)
MEDIAPIPE_UPPER_LIDS = ((246, 161, 160, 159, 158, 157, 173), (466, 388, 387, 386, 385, 384, 398))
MEDIAPIPE_EYE_CORNERS = ((33, 133), (362, 263))
DARKEST = 0.3
"""The colour of hair is the median of the darkest share of the pixels sampled at it."""

GROUPS = ("lashes", "brows")


@dataclass(frozen=True)
class FaceHair:
    faces: np.ndarray
    """``(S,)`` the triangle each strand is bound to."""
    barycentric: np.ndarray
    """``(S, 3)`` where on it the root is."""
    points: np.ndarray
    """``(S, POINTS, 3)`` the strand's points in metres from the root, in the triangle's frame."""
    widths: np.ndarray
    """``(S,)`` metres across at the root."""
    groups: np.ndarray
    """``(S,)`` index into ``GROUPS``."""


def grow_face_hair(
    avatar: Avatar,
    textures: TextureRecord,
    ict_directory,
    identity: np.ndarray,
    rng: np.random.Generator,
) -> FaceHair:
    """Grow the lashes and brows on the avatar's face.

    Args:
        identity: ``(I,)`` the ICT-FaceKit identity the avatar's head was built with.
    """
    model, model_faces = ict.all_vertices(ict_directory, identity)
    head = avatar.rest_vertices[avatar.head_start : avatar.head_start + ict.FACE.stop]
    scale, rotation, shift = (
        np.asarray(x, np.float64)
        for x in _umeyama(torch.from_numpy(model[: ict.FACE.stop]).float(), torch.from_numpy(head))
    )
    to_avatar = lambda points: scale * points @ rotation.T + shift  # noqa: E731

    skin = model[ict.SKIN]
    cards = [to_avatar(grid) for grid in lash_cards(model, model_faces, skin)]
    eyes = avatar.rest_vertices[np.unique(avatar.faces[avatar.face_parts == EYES])]
    lashes = [_lashes(card, _lid(card, eyes), rng) for card in cards]
    landmarks = avatar.head_start + np.asarray(ict.landmark_vertices(ict_directory))
    brows = [
        _brows(avatar, textures, avatar.rest_vertices[landmarks[list(line)]], rng)
        for line in BROW_LANDMARKS
    ]
    strands = np.concatenate(lashes + brows)
    groups = np.repeat(
        [0] * len(lashes) + [1] * len(brows), [len(s) for s in lashes + brows]
    ).astype(np.uint8)
    widths = np.where(groups == 0, LASH_WIDTH, BROW_WIDTH)
    return _bind(avatar, strands, widths, groups)


def lash_cards(vertices: np.ndarray, faces: np.ndarray, skin: np.ndarray) -> list[np.ndarray]:
    """ICT-FaceKit's eyelash cards as ``(rows, columns, 3)`` grids, rows from the roots (the
    vertices touching the ``(V, 3)`` skin) to the tips, columns along the lid."""
    first, last = ict.LASH_CARDS.start, ict.LASH_CARDS.stop
    inside = (faces.min(axis=1) >= first) & (faces.max(axis=1) < last)
    triangles = faces[inside] - first
    points = vertices[ict.LASH_CARDS]
    edges = np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
    graph = sparse.coo_matrix((np.ones(len(edges)), edges.T), shape=(len(points),) * 2).tocsr()
    graph = graph + graph.T
    count, labels = connected_components(graph, directed=False)
    near_skin = cKDTree(skin).query(points)[0] < ROOT_TOLERANCE

    grids = []
    for card in range(count):
        members = np.flatnonzero(labels == card)
        if not near_skin[members].any():
            continue  # loose vertices, not a card
        rows = _rows(graph, members, near_skin)
        roots = points[rows[0]]
        axis = np.linalg.svd(roots - roots.mean(axis=0))[2][0]
        columns = [rows[0][np.argsort((roots - roots.mean(axis=0)) @ axis)]]
        for row in rows[1:]:
            previous = points[columns[-1]]
            cost = np.linalg.norm(previous[:, None] - points[row][None], axis=-1)
            _, match = linear_sum_assignment(cost)
            columns.append(row[match])
        grids.append(points[np.stack(columns)])
    return grids


def _rows(graph: sparse.csr_matrix, members: np.ndarray, near_skin: np.ndarray) -> list[np.ndarray]:
    """A card's vertices in rows by how many edges they are from its roots."""
    roots = members[near_skin[members]]
    steps = shortest_path(graph, indices=roots, unweighted=True).min(axis=0)[members]
    return [members[steps == step] for step in np.unique(steps)]


def _lid(card: np.ndarray, eyes: np.ndarray) -> str:
    centre = eyes[np.linalg.norm(eyes - card[0].mean(axis=0), axis=1) < 0.02].mean(axis=0)
    return "upper" if card[0, :, 2].mean() > centre[2] else "lower"


def _lashes(card: np.ndarray, lid: str, rng: np.random.Generator) -> np.ndarray:
    """``(N, POINTS, 3)`` lashes interpolated along a ``(rows, columns, 3)`` card."""
    rows, columns, _ = card.shape
    count = LASHES_PER_CARD[lid]
    along = (np.arange(count) + rng.uniform(0.15, 0.85, count)) / count * (columns - 1)
    left = np.clip(along.astype(int), 0, columns - 2)
    share = (along - left)[:, None, None]
    lashes = (1 - share) * card[:, left].transpose(1, 0, 2) + share * card[:, left + 1].transpose(
        1, 0, 2
    )
    lashes = _resample(lashes, POINTS)
    offsets = (lashes - lashes[:, :1]) * rng.uniform(*LASH_LENGTH, count)[:, None, None]
    tangent = card[0, -1] - card[0, 0]
    tangent /= np.linalg.norm(tangent)
    turned = np.stack([_rotation(axis, rng.normal(0, LASH_SPREAD)) for axis in (tangent,) * count])
    return lashes[:, :1] + np.einsum("nij,npj->npi", turned, offsets)


def _brows(
    avatar: Avatar, textures: TextureRecord, line: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """``(N, POINTS, 3)`` brow hairs along the ``(5, 3)`` landmarks of one brow."""
    if abs(line[0, 0]) > abs(line[-1, 0]):
        line = line[::-1]  # the inner end first
    skin = np.flatnonzero(avatar.face_parts == HEAD_SKIN)
    corners = avatar.rest_vertices[avatar.faces[skin]]
    centres = corners.mean(axis=1)
    near = np.min(np.linalg.norm(centres[:, None] - line[None], axis=-1), axis=1) < 0.02
    faces = skin[near]
    corners = corners[near]

    area = np.linalg.norm(
        np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1
    )
    picked = rng.choice(len(faces), BROW_CANDIDATES, p=area / area.sum())
    weights = rng.dirichlet(np.ones(3), BROW_CANDIDATES)
    points = np.einsum("nc,ncd->nd", weights, corners[picked])
    along, height, tangent = _along(line, points)
    band = (height > BROW_BAND[0]) & (height < BROW_BAND[1]) & (along > 0) & (along < 1)

    brightness = texture_at(textures, avatar, faces[picked], weights).sum(axis=1)
    surrounding = np.quantile(brightness[band | (height > 0)], 0.8)
    darkness = np.clip((surrounding - brightness) / (BROW_CONTRAST * surrounding), 0, 1) * band
    darkness = np.where(darkness > BROW_MIN_DARKNESS, darkness, 0)
    count = min(BROW_HAIRS, int((darkness > 0).sum()))
    if count == 0:
        return np.zeros((0, POINTS, 3))
    chosen = rng.choice(len(points), count, replace=False, p=darkness / darkness.sum())

    normals = triangle_frames(
        torch.from_numpy(avatar.rest_vertices), torch.from_numpy(avatar.faces[faces[picked]])
    )[0][..., 2].numpy()
    return np.stack(
        [_brow_hair(points[i], normals[i], tangent[i], along[i], height[i], rng) for i in chosen]
    )


def _along(line: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, ...]:
    """For ``(N, 3)`` points near a polyline: where along it they lie, 0 at its start and 1 at
    its end; their height above it, in metres; and the line's unit direction there."""
    starts, ends = line[:-1], line[1:]
    segments = ends - starts
    lengths = np.linalg.norm(segments, axis=1)
    t = np.clip(
        np.einsum("nsd,sd->ns", points[:, None] - starts[None], segments) / lengths**2, 0, 1
    )
    nearest = starts[None] + t[..., None] * segments[None]
    segment = np.linalg.norm(points[:, None] - nearest, axis=-1).argmin(axis=1)
    rows = np.arange(len(points))
    travelled = np.concatenate(([0], np.cumsum(lengths)))
    along = (travelled[segment] + t[rows, segment] * lengths[segment]) / travelled[-1]
    height = points[:, 2] - nearest[rows, segment, 2]
    # Before its start or past its end the line goes on straight.
    along = np.where((segment == 0) & (t[rows, 0] == 0), -1.0, along)
    along = np.where((segment == len(segments) - 1) & (t[rows, -1] == 1), 2.0, along)
    return along, height, segments[segment] / lengths[segment, None]


def _brow_hair(
    root: np.ndarray,
    normal: np.ndarray,
    tangent: np.ndarray,
    along: float,
    height: float,
    rng: np.random.Generator,
) -> np.ndarray:
    def flat(vector: np.ndarray) -> np.ndarray:
        vector = vector - normal * (vector @ normal)
        return vector / np.linalg.norm(vector)

    up = flat(np.array([0.0, 0, 1]))
    outward = flat(tangent)
    inner = 1 - _smoothstep(0.1, 0.35, along)
    middle = (BROW_BAND[0] + BROW_BAND[1]) / 2
    converge = np.clip((height - middle) / 0.005, -1, 1)
    direction = flat(
        (0.75 * up + 0.35 * outward) * inner + outward * (1 - inner) - 0.45 * converge * up
    )
    direction = _rotation(normal, rng.normal(0, 0.12)) @ direction
    length = np.interp(along, (0, 0.5, 1), BROW_LENGTH) * rng.uniform(0.8, 1.15)
    lift = BROW_LIFT[1] + (BROW_LIFT[0] - BROW_LIFT[1]) * inner
    s = np.linspace(0, 1, POINTS)[:, None]
    # Rising off the skin, then curving back down to it towards the tip.
    return root + length * (s * direction + np.sin(lift) * (s - 0.8 * s**2) * normal)


def _bind(avatar: Avatar, strands: np.ndarray, widths: np.ndarray, groups: np.ndarray) -> FaceHair:
    skin = np.flatnonzero(avatar.face_parts == HEAD_SKIN)
    centres = avatar.rest_vertices[avatar.faces[skin]].mean(axis=1)
    near = cKDTree(strands[:, 0]).query(centres)[0] < 0.01
    candidates = skin[near]
    vertices = torch.from_numpy(avatar.rest_vertices).double()
    found, barycentric = closest_points(
        torch.from_numpy(strands[:, 0]).double(),
        vertices,
        torch.from_numpy(avatar.faces[candidates]),
    )
    faces = candidates[found.numpy()]
    frames = triangle_frames(vertices, torch.from_numpy(avatar.faces[faces]))[0].numpy()
    roots = np.einsum("sc,scd->sd", barycentric.numpy(), avatar.rest_vertices[avatar.faces[faces]])
    local = np.einsum("sji,spj->spi", frames, strands - roots[:, None])
    return FaceHair(
        faces=faces,
        barycentric=barycentric.numpy(),
        points=local,
        widths=widths,
        groups=groups,
    )


def measure_hair_colours(frames: np.ndarray, landmarks: np.ndarray) -> dict[str, np.ndarray]:
    """The ``(3,)`` RGB in [0, 1] of the brows and of the lashes along the upper lids, in the
    frames that face the camera.

    Args:
        frames: ``(N, H, W, 3)`` uint8 RGB.
        landmarks: ``(N, 478, 3)`` MediaPipe face landmarks, x, y in pixels; NaN where none.
    """
    samples = {"brows": [], "lashes": []}
    for frame, points in zip(frames, landmarks, strict=True):
        if np.isnan(points).any() or not frontal(points):
            continue
        for brow, lid, corners in zip(
            MEDIAPIPE_BROWS, MEDIAPIPE_UPPER_LIDS, MEDIAPIPE_EYE_CORNERS, strict=True
        ):
            width = np.linalg.norm(points[corners[0], :2] - points[corners[1], :2])
            samples["brows"].append(_pixels(frame, points[list(brow), :2], 0.05 * width))
            above = points[list(lid), :2] - [0, 0.02 * width]
            samples["lashes"].append(_pixels(frame, above, 0.02 * width))
    if not samples["brows"]:
        raise ValueError("no frame faces the camera")
    colours = {}
    for name, pixels in samples.items():
        pixels = np.concatenate(pixels)
        brightness = pixels.sum(axis=1)
        darkest = pixels[brightness <= np.quantile(brightness, DARKEST)]
        colours[name] = np.median(darkest, axis=0) / 255
    return colours


def face_hair_metadata(colours: dict[str, np.ndarray]) -> dict:
    """The face hair's entry in the avatar file's metadata."""
    shades = {"lashes": 1.0, "brows": BROW_SHADE}
    return {
        "points_per_strand": POINTS,
        "groups": [
            {"name": name, "colour": (colours[name] * shades[name]).tolist()} for name in GROUPS
        ],
    }


def _pixels(frame: np.ndarray, centres: np.ndarray, radius: float) -> np.ndarray:
    ys, xs = np.mgrid[: frame.shape[0], : frame.shape[1]]
    taken = np.zeros(frame.shape[:2], bool)
    low = np.maximum((centres.min(axis=0) - radius).astype(int), 0)
    high = np.minimum((centres.max(axis=0) + radius).astype(int) + 1, frame.shape[1::-1])
    window = (slice(low[1], high[1]), slice(low[0], high[0]))
    for x, y in centres:
        taken[window] |= (xs[window] - x) ** 2 + (ys[window] - y) ** 2 <= radius**2
    return frame[taken].astype(np.float64)


def texture_at(
    textures: TextureRecord, avatar: Avatar, faces: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """``(N, 3)`` the head texture's colour at points on head triangles."""
    assert (avatar.face_textures[faces] == HEAD_TEXTURE).all()
    uv = np.einsum("nc,ncd->nd", weights, avatar.uvs[faces])
    height, width = textures.head.shape[:2]
    x = np.clip((uv[:, 0] * width).astype(int), 0, width - 1)
    y = np.clip(((1 - uv[:, 1]) * height).astype(int), 0, height - 1)
    return textures.head[y, x]


def _resample(polylines: np.ndarray, count: int) -> np.ndarray:
    """``(N, count, 3)`` points evenly spaced along ``(N, P, 3)`` polylines."""
    steps = np.linalg.norm(np.diff(polylines, axis=1), axis=-1)
    travelled = np.concatenate([np.zeros((len(polylines), 1)), np.cumsum(steps, axis=1)], axis=1)
    targets = np.linspace(0, 1, count)[None] * travelled[:, -1:]
    return np.stack(
        [
            np.stack([np.interp(target, done, line[:, d]) for d in range(3)], axis=-1)
            for target, done, line in zip(targets, travelled, polylines, strict=True)
        ]
    )


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """``(3, 3)`` rotation by ``angle`` radians about a unit ``axis``."""
    x, y, z = axis
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * cross @ cross


def _smoothstep(low: float, high: float, x: float) -> float:
    t = np.clip((x - low) / (high - low), 0, 1)
    return t * t * (3 - 2 * t)
