"""The inside of the mouth, for the runtime's mouth materials: stock teeth, gums, tongue and the
mouth's socket, and the lips' wet inner side.

The videos see the inside of the mouth rarely and badly: baked from them, the teeth carry bits
of lips, the gums bits of teeth and the lips' inner side bits of teeth. The runtime draws these
surfaces with fixed materials instead, from the regions tagged here: their baked texture is
unused and they get no Gaussians. The inner lips take the colour of the lips just outside them.

The lower teeth also come forward as the jaw opens. The performer's lower lip is full, and
ICT-FaceKit's ``jawOpen`` moves the lower teeth exactly as far as the lip: their edges stay at
its rim, hidden behind it, where an open mouth shows them.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import torch
from scipy.spatial import cKDTree

from avatar_splats.body_models import ict
from avatar_splats.body_models.hybrid import Avatar
from avatar_splats.body_models.mesh import boundary_loops, triangle_frames
from avatar_splats.export.face_hair import texture_at
from avatar_splats.training.textures import TextureRecord

SOCKET, GUMS, TONGUE, TEETH, INNER_LIPS = 1, 2, 3, 4, 5
"""Values of the ``mouth_regions`` array; 0 is outside the mouth."""
REGION_NAMES = ("none", "socket", "gums", "tongue", "teeth", "inner_lips")

MOUTH_SOCKET = slice(11248, 13294)
EYE_SOCKETS = slice(13294, 14062)
GUMS_AND_TONGUE = slice(14062, 17039)
UPPER_TEETH = slice(17039, 19247)
LOWER_TEETH = slice(19247, 21451)
"""ICT-FaceKit's vertex ranges, from its README."""
FACE_END = ict.FACE.stop

INNER_LIP_REACH = 0.008
"""How far, in metres, from the skin's edge at the mouth the lips' inner side may reach."""
FACING_FORWARD = 0.35
"""Lip triangles whose normal points forward (along -y, where the face looks) by less than this
share face up, down or back, into the mouth or onto the other lip: they are the inner side."""
TONGUE_DISTANCE = 0.004
"""Gum and tongue vertices further than this, in metres, from every tooth are tongue."""
LOWER_JAW_LIFT = np.array([0.0, -0.001, 0.002])
"""How far the lower teeth, and the gums and tongue below the bite, move up (+z) and forward
(-y) with the jaw fully open, in metres, on top of ``jawOpen``."""
BITE_BLEND = 0.002
"""Over how many metres below the bite the gums and tongue come to follow the lower teeth."""

COLOURS = {
    "teeth": [0.88, 0.85, 0.78],
    "gums": [0.80, 0.46, 0.46],
    "tongue": [0.82, 0.50, 0.50],
    "socket": [0.42, 0.17, 0.18],
}
"""RGB in [0, 1], as the videos' colours: stock values, not measured."""
INNER_LIP_SHADE = 0.95
"""The lips' inner side is this much darker than their outer side."""


def mouth_regions(avatar: Avatar) -> np.ndarray:
    """``(F,)`` uint8: each triangle's region inside the mouth, 0 outside it, and everywhere on
    a head that is not ICT-FaceKit's."""
    head = avatar.faces - avatar.head_start
    first = head.min(axis=1)
    inside = (avatar.faces >= avatar.head_start).all(axis=1)
    regions = np.zeros(len(avatar.faces), np.uint8)
    if len(avatar.rest_vertices) - avatar.head_start < LOWER_TEETH.stop:
        return regions
    for span, region in (
        (MOUTH_SOCKET, SOCKET),
        (EYE_SOCKETS, SOCKET),
        (GUMS_AND_TONGUE, GUMS),
        (UPPER_TEETH, TEETH),
        (LOWER_TEETH, TEETH),
    ):
        regions[inside & (first >= span.start) & (first < span.stop)] = region

    teeth = avatar.rest_vertices[avatar.head_start + np.r_[UPPER_TEETH, LOWER_TEETH]]
    gums = np.flatnonzero(regions == GUMS)
    centres = avatar.rest_vertices[avatar.faces[gums]].mean(axis=1)
    far = cKDTree(teeth).query(centres)[0] > TONGUE_DISTANCE
    regions[gums[far]] = TONGUE

    regions[_inner_lips(avatar)[0]] = INNER_LIPS
    return regions


def _inner_lips(avatar: Avatar) -> tuple[np.ndarray, np.ndarray]:
    """The lips' inner side, and their outer side right next to it: two arrays of triangles."""
    skin, distance = _from_mouth_edge(avatar)
    near = skin[distance < INNER_LIP_REACH]
    normals = triangle_frames(
        torch.from_numpy(avatar.rest_vertices).double(), torch.from_numpy(avatar.faces[near])
    )[0][..., 2].numpy()
    inner = -normals[:, 1] < FACING_FORWARD
    return near[inner], near[~inner]


def _from_mouth_edge(avatar: Avatar) -> tuple[np.ndarray, np.ndarray]:
    """The face's triangles and how far, in metres, their nearest corner is from the skin's edge
    around the mouth's opening, where the lips turn inwards to meet the mouth's socket."""
    skin = np.flatnonzero(
        (avatar.faces >= avatar.head_start).all(axis=1)
        & (avatar.faces < avatar.head_start + FACE_END).all(axis=1)
    )
    front_teeth = avatar.rest_vertices[avatar.head_start + np.r_[UPPER_TEETH]]
    middle = front_teeth[front_teeth[:, 1] < front_teeth[:, 1].min() + 0.002].mean(axis=0)
    loops = boundary_loops(avatar.faces[skin])
    mouth = min(loops, key=lambda loop: np.linalg.norm(avatar.rest_vertices[loop].mean(0) - middle))
    tree = cKDTree(avatar.rest_vertices[mouth])
    corners = avatar.rest_vertices[avatar.faces[skin]]
    distance = tree.query(corners.reshape(-1, 3))[0].reshape(-1, 3).min(axis=1)
    return skin, distance


def expose_lower_teeth(avatar: Avatar) -> Avatar:
    """The avatar with ``jawOpen`` bringing the lower teeth, and the gums and tongue below the
    bite, up and forward by ``LOWER_JAW_LIFT``."""
    names = list(avatar.expression_names)
    if "jawOpen" not in names:
        return avatar
    head = avatar.rest_vertices[avatar.head_start :]
    index = np.arange(len(head))
    bite = head[LOWER_TEETH, 2].max()
    weights = np.zeros(len(head))
    weights[LOWER_TEETH] = 1
    soft = (index >= GUMS_AND_TONGUE.start) & (index < GUMS_AND_TONGUE.stop)
    weights[soft] = np.clip((bite - head[soft, 2]) / BITE_BLEND, 0, 1)
    basis = avatar.expression_basis.copy()
    basis[names.index("jawOpen")] += weights[:, None] * LOWER_JAW_LIFT
    return dataclasses.replace(avatar, expression_basis=basis)


def mouth_metadata(avatar: Avatar, textures: TextureRecord) -> dict:
    """The mouth's entry in the avatar file's metadata: the regions' names and colours."""
    colours = dict(COLOURS)
    colours["inner_lips"] = [0.7, 0.4, 0.4]
    if len(avatar.rest_vertices) - avatar.head_start >= LOWER_TEETH.stop:
        _, outer = _inner_lips(avatar)
        lips = texture_at(textures, avatar, outer, np.full((len(outer), 3), 1 / 3))
        colours["inner_lips"] = (np.median(lips, axis=0) * INNER_LIP_SHADE).tolist()
    return {"regions": list(REGION_NAMES), "colours": colours}
