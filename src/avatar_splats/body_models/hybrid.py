"""The avatar's mesh: the fitted Anny body with its head replaced by the fitted ICT-FaceKit head.

The head is placed on the body by aligning the two heads. Both meshes are then cut at a seam
``neck_seam`` below the chin: the head model keeps the head and upper neck (its lower neck flares
out into a collar), Anny keeps the rest. The head model's cut edge is moved onto Anny's neck, its
neck bent to follow over ``neck_blend``, and a band of triangles closes the gap, so the avatar is
one surface: drawn as a mesh, it neither cracks nor shows two necks fighting. The head's vertices
take their skinning weights from Anny's nearest vertices, so the joined neck bends as one.

Anny has no genitals; generated ones (``genitals``) replace its crotch. Their vertices come after
the head's, so the blendshapes, which move every vertex from ``head_start`` on, carry their
shapes next to the head's expressions.

The mesh carries two textures: the body's, in Anny's UV layout (which the bands at the neck and
the crotch also use), and an atlas: ICT-FaceKit's tiles for the head, and one more for the
genitals.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import roma
import scipy.sparse
import scipy.sparse.csgraph
import torch

from avatar_splats.body_models import ict
from avatar_splats.body_models.anny import AnnyBody
from avatar_splats.body_models.genitals import build_insert
from avatar_splats.body_models.ict import IctFaceKit
from avatar_splats.body_models.mesh import (
    boundary_loops,
    closest_points,
    loop_subdivision,
    plane_barycentric,
    uniform_laplacian,
    unique_edges,
    zip_loops,
)
from avatar_splats.body_models.skinning import linear_blend_skinning, sparsify_weights
from avatar_splats.config import BodyModelsConfig
from avatar_splats.records import ArrayRecord

logger = logging.getLogger(__name__)

BODY, HEAD_SKIN, MOUTH, EYES, HANDS, GENITALS, CANALS = 0, 1, 2, 3, 4, 5, 6
"""Parts of the avatar's mesh, as stored in ``Avatar.face_parts``: ``CANALS`` are the walls of
the vagina and the anal canal, which no video sees."""
PART_NAMES = ("body", "head_skin", "mouth", "eyes", "hands", "genitals", "canals")
"""The parts' names, in the order of their numbers."""
HEAD_PARTS = (HEAD_SKIN, MOUTH, EYES)
BODY_TEXTURE, HEAD_TEXTURE = 0, 1
"""The avatar's textures, as stored in ``Avatar.face_textures``: the head's is the atlas."""
ATLAS_TILES = ict.UV_TILES + 1
"""Square tiles side by side in the atlas: ICT-FaceKit's, then the genitals'."""

ICT_TO_WORLD = torch.tensor([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])
"""ICT-FaceKit's axes (y up, facing +z) to Anny's (z up, facing -y)."""

COCO_NOSE, COCO_LEFT_EYE, COCO_RIGHT_EYE = 0, 1, 2
ICT_NOSE_TIP, ICT_RIGHT_EYE, ICT_LEFT_EYE = 30, (36, 39), (42, 45)
"""Multi-PIE landmarks matching Anny's COCO nose and eye keypoints."""
ICT_CHIN = 8
MATCHED_BELOW_EYES = 0.14
"""How far below the eyes, in metres, the head surfaces are matched to place the head."""
SEAM_SMOOTHING = 10
"""Smoothing steps for the vertices along the seam."""
NECK_SURFACE = 0.05
"""How far above and below the seam, in metres, Anny's neck is searched for the nearest surface."""


@dataclass(frozen=True)
class Avatar(ArrayRecord):
    """A skinned mesh with expression blendshapes on its head: the scaffold Gaussians bind to.

    Coordinates are Anny's: metres, z up, facing -y.
    """

    rest_vertices: np.ndarray
    """``(V, 3)`` vertices in the rest pose with a neutral face."""
    faces: np.ndarray
    """``(F, 3)`` triangles."""
    face_parts: np.ndarray
    """``(F,)`` each triangle's part: ``BODY``, ``HANDS``, ``HEAD_SKIN``, ``MOUTH``, ``EYES``,
    ``GENITALS`` or ``CANALS``."""
    uvs: np.ndarray
    """``(F, 3, 2)`` texture coordinates of the triangles' corners, in [0, 1] with v up."""
    face_textures: np.ndarray
    """``(F,)`` each triangle's texture: ``BODY_TEXTURE`` or ``HEAD_TEXTURE``."""
    bone_indices: np.ndarray
    """``(V, K)`` the bones each vertex is bound to."""
    bone_weights: np.ndarray
    """``(V, K)`` skinning weights, summing to 1 per vertex."""
    bone_labels: np.ndarray
    """``(J,)`` bone names."""
    bone_parents: np.ndarray
    """``(J,)`` index of each bone's parent; -1 for the root."""
    bind_poses: np.ndarray
    """``(J, 4, 4)`` bone frames in the rest pose."""
    head_start: int
    """Index of the first head vertex. The head's vertices follow, then the genitals'."""
    expression_names: np.ndarray
    """``(E,)`` ARKit-style expression names, then the genitals' shapes."""
    expression_basis: np.ndarray
    """``(E, V - head_start, 3)`` offsets of the vertices from the head's on, at weight 1."""


class Rig(torch.nn.Module):
    """Poses an avatar: expression blendshapes on the rest pose, then linear blend skinning."""

    def __init__(self, avatar: Avatar):
        super().__init__()
        self.head_start = avatar.head_start
        for name in ("rest_vertices", "bone_weights", "bind_poses", "expression_basis"):
            self.register_buffer(name, torch.from_numpy(getattr(avatar, name)).float())
        for name in ("faces", "face_parts", "bone_indices"):
            self.register_buffer(name, torch.from_numpy(getattr(avatar, name)).long())

    def forward(
        self,
        skinning: torch.Tensor,
        expressions: torch.Tensor | None = None,
        body_offsets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``(B, V, 3)`` vertices for ``(B, J, 4, 4)`` skinning, ``(B, E)`` expressions and
        ``(B, head_start, 3)`` offsets of the body's rest vertices (body shapes)."""
        rest = self.rest_vertices.expand(len(skinning), -1, -1)
        if body_offsets is not None:
            rest = torch.cat(
                (rest[:, : self.head_start] + body_offsets, rest[:, self.head_start :]), 1
            )
        if expressions is not None:
            offsets = torch.einsum("be,evc->bvc", expressions, self.expression_basis)
            rest = torch.cat((rest[:, : self.head_start], rest[:, self.head_start :] + offsets), 1)
        return linear_blend_skinning(rest, self.bone_indices, self.bone_weights, skinning)


@dataclass(frozen=True)
class HeadPlacement:
    """The similarity transform putting the head model's vertices on the body in its rest pose."""

    scale: float
    rotation: torch.Tensor
    """``(3, 3)``, including the change from the head model's axes to Anny's."""
    translation: torch.Tensor
    """``(3,)`` in metres."""

    def apply(self, points: torch.Tensor) -> torch.Tensor:
        return self.scale * points @ self.rotation.T + self.translation


def rest_body(
    body: AnnyBody, phenotype: np.ndarray, local_changes: np.ndarray
) -> dict[str, torch.Tensor]:
    """The fitted body in its rest pose: ``vertices`` ``(V, 3)``, ``keypoints`` ``(17, 3)`` and
    ``bind_poses`` ``(J, 4, 4)``."""
    device = body.faces.device
    out = body(
        torch.from_numpy(phenotype)[None].to(device),
        torch.from_numpy(local_changes)[None].to(device),
        torch.zeros(1, body.num_bones, 3, device=device),
        torch.zeros(1, 3, device=device),
    )
    vertices = out["rest_vertices"][0]
    return {
        "vertices": vertices,
        "keypoints": body.keypoint_regressor({"vertices": vertices[None]})[0],
        "bind_poses": out["rest_bone_poses"][0],
    }


def rest_head(head: IctFaceKit, identity: np.ndarray) -> torch.Tensor:
    """``(V, 3)`` the fitted head with a neutral face, in the head model's own coordinates."""
    zeros = head.expression_basis.new_zeros(head.num_expressions)
    return head(torch.from_numpy(identity).to(zeros.device), zeros)


def build_avatar(
    body: AnnyBody,
    body_rest: dict[str, torch.Tensor],
    head: IctFaceKit,
    head_rest: torch.Tensor,
    placement: HeadPlacement,
    config: BodyModelsConfig,
) -> Avatar:
    """Join the fitted body and head, placed on it, into one rigged, textured mesh."""
    device = body.faces.device
    bind_poses = body_rest["bind_poses"]
    body_rest_all, body_rest = body_rest, body_rest["vertices"]
    head = head.to(device)
    scale, rotation = placement.scale, placement.rotation
    head_world = placement.apply(head_rest)
    seam = head_world[head.landmarks[ICT_CHIN], 2] - config.neck_seam

    insert = build_insert(body, body_rest_all, config.genitals)
    body_rest = body_rest.clone()
    body_rest[insert.border] = insert.border_vertices
    body_kept = ~_above(body.faces, body_rest, seam) & ~insert.removed
    body_kept &= ~body.head_faces | (body_rest[body.faces][..., 2].amax(dim=1) <= seam)
    head_kept = head_world[head.faces][..., 2].amin(dim=1) >= seam
    body_faces, head_faces = body.faces[body_kept], head.faces[head_kept]
    neck = _neck_surface(body, body_rest, seam)
    body_loop = _loop_at(body_faces, body_rest, seam)
    head_loop = _loop_at(head_faces, head_world, seam)
    head_world = _bend_neck(head_world, head_loop, body_rest, body.faces[neck], config.neck_blend)

    kept = torch.unique(body_faces)
    remap = torch.full((len(body_rest),), -1, dtype=torch.long, device=device)
    remap[kept] = torch.arange(len(kept), device=device)
    bridge = torch.from_numpy(
        zip_loops(body_rest[body_loop].cpu().numpy(), head_world[head_loop].cpu().numpy())
    ).to(device)
    ends = torch.cat((remap[body_loop], len(kept) + head_loop))
    bridge_vertices = torch.cat((body_rest[body_loop], head_world[head_loop]))[bridge]
    bridge_uvs = _surface_uvs(bridge_vertices, body, body_rest, neck)

    insert_start = len(kept) + len(head_world)
    insert_faces = torch.where(
        insert.faces < len(body_rest),
        remap[insert.faces.clamp_max(len(body_rest) - 1)],
        insert.faces - len(body_rest) + insert_start,
    )
    if (insert_faces < 0).any():
        raise ValueError("the genitals are stitched to vertices of Anny's that were cut away")
    vertices = torch.cat((body_rest[kept], head_world, insert.vertices))
    faces = torch.cat((remap[body_faces], ends[bridge], head_faces + len(kept), insert_faces))
    vertices = _smooth_seam(vertices, faces, ends)

    k = config.max_bone_influences
    body_bones, body_weights = sparsify_weights(body.skin_weights[kept], k)
    head_bones, head_weights = sparsify_weights(transfer_weights(body, body_rest, head_world), k)
    insert_bones, insert_weights = sparsify_weights(insert.weights, k)

    parts = torch.full((len(head_faces),), HEAD_SKIN, device=device)
    parts[head_faces.min(dim=1).values >= ict.MOUTH_INTERIOR.start] = MOUTH
    parts[head_faces.min(dim=1).values >= ict.EYEBALLS.start] = EYES
    on_body = len(body_faces) + len(bridge)
    body_parts = torch.full((on_body,), BODY, device=device)
    body_parts[: len(body_faces)][body.hand_faces[body_kept]] = HANDS

    expressions = scale * head.expression_basis @ rotation.T
    count = len(head_world) + len(insert.vertices)
    expression_basis = torch.zeros(len(expressions) + len(insert.shapes), count, 3, device=device)
    expression_basis[: len(expressions), : len(head_world)] = expressions
    expression_basis[len(expressions) :, len(head_world) :] = insert.shapes
    insert_parts = torch.where(insert.inside, CANALS, GENITALS)
    insert_parts[insert.on_body_texture] = BODY
    head_uvs = head.uvs[head_kept].clone()
    head_uvs[..., 0] *= ict.UV_TILES / ATLAS_TILES
    insert_uvs = insert.uvs.clone()
    atlas = ~insert.on_body_texture
    insert_uvs[atlas, :, 0] = (ict.UV_TILES + insert_uvs[atlas, :, 0]) / ATLAS_TILES
    logger.info(
        "avatar: %d body, %d head and %d genital vertices, head scaled by %.3f, "
        "%d triangles at the neck",
        len(kept),
        len(head_world),
        len(insert.vertices),
        scale,
        len(bridge),
    )
    return Avatar(
        rest_vertices=vertices.cpu().numpy(),
        faces=faces.cpu().numpy(),
        face_parts=torch.cat((body_parts, parts, insert_parts)).to(torch.int8).cpu().numpy(),
        uvs=torch.cat((body.uvs[body_kept], bridge_uvs, head_uvs, insert_uvs)).cpu().numpy(),
        face_textures=np.concatenate(
            (
                np.repeat(
                    np.array([BODY_TEXTURE, HEAD_TEXTURE], np.int8), [on_body, len(head_faces)]
                ),
                np.where(insert.on_body_texture.cpu().numpy(), BODY_TEXTURE, HEAD_TEXTURE),
            )
        ).astype(np.int8),  # fmt: skip
        bone_indices=torch.cat((body_bones, head_bones, insert_bones)).cpu().numpy(),
        bone_weights=torch.cat((body_weights, head_weights, insert_weights)).cpu().numpy(),
        bone_labels=np.array(body.bone_labels),
        bone_parents=np.array(body.bone_parents),
        bind_poses=bind_poses.cpu().numpy(),
        head_start=len(kept),
        expression_names=np.array(head.expression_names + insert.shape_names),
        expression_basis=expression_basis.cpu().numpy(),
    )


def subdivide(avatar: Avatar, max_bone_influences: int) -> Avatar:
    """The avatar with each triangle split in four and its surface smoothed (Loop subdivision):
    positions, skinning weights and expression blendshapes all follow the same smoothing, texture
    coordinates are split linearly. New vertices between head vertices are head vertices."""
    faces = torch.from_numpy(avatar.faces).long()
    num_vertices = len(avatar.rest_vertices)
    split, matrix = loop_subdivision(faces, num_vertices)

    head_start = avatar.head_start
    edges, _ = unique_edges(faces)
    on_head = torch.cat((torch.arange(num_vertices) >= head_start, (edges >= head_start).all(1)))
    order = torch.cat((torch.nonzero(~on_head).squeeze(-1), torch.nonzero(on_head).squeeze(-1)))
    position = torch.empty_like(order)
    position[order] = torch.arange(len(order))

    def smoothed(values: torch.Tensor) -> torch.Tensor:
        return (matrix @ values.reshape(num_vertices, -1)).reshape(-1, *values.shape[1:])[order]

    dense = torch.zeros(num_vertices, len(avatar.bone_labels))
    dense.scatter_(
        1, torch.from_numpy(avatar.bone_indices).long(), torch.from_numpy(avatar.bone_weights)
    )
    bone_indices, bone_weights = sparsify_weights(smoothed(dense), max_bone_influences)
    basis = torch.from_numpy(avatar.expression_basis)
    full_basis = torch.zeros(num_vertices, len(basis), 3)
    full_basis[head_start:] = basis.permute(1, 0, 2)
    new_head_start = int((~on_head).sum())

    uvs = torch.from_numpy(avatar.uvs)
    a, b, c = uvs.unbind(1)
    ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
    split_uvs = torch.stack(
        (
            torch.stack((a, ab, ca), 1),
            torch.stack((ab, b, bc), 1),
            torch.stack((ca, bc, c), 1),
            torch.stack((ab, bc, ca), 1),
        ),
        dim=1,
    ).reshape(-1, 3, 2)
    return Avatar(
        rest_vertices=smoothed(torch.from_numpy(avatar.rest_vertices)).numpy(),
        faces=position[split].numpy(),
        face_parts=np.repeat(avatar.face_parts, 4),
        uvs=split_uvs.numpy(),
        face_textures=np.repeat(avatar.face_textures, 4),
        bone_indices=bone_indices.numpy(),
        bone_weights=bone_weights.numpy(),
        bone_labels=avatar.bone_labels,
        bone_parents=avatar.bone_parents,
        bind_poses=avatar.bind_poses,
        head_start=new_head_start,
        expression_names=avatar.expression_names,
        expression_basis=smoothed(full_basis)[new_head_start:].permute(1, 0, 2).numpy(),
    )


def _above(faces: torch.Tensor, vertices: torch.Tensor, seam: torch.Tensor) -> torch.Tensor:
    """``(F,)`` bool: the triangles reaching above the seam that hang together with the top of
    the head. Anny's head part ends near the chin, so the cut cannot follow the parts alone; its
    eyes, which hang together with nothing, go with the head part."""
    above = vertices[faces][..., 2].amax(dim=1) > seam
    edges = faces[above][:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2).cpu().numpy()
    graph = scipy.sparse.coo_matrix(
        (np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(len(vertices),) * 2
    )
    _, labels = scipy.sparse.csgraph.connected_components(graph, directed=False)
    labels = torch.from_numpy(labels).to(faces.device)
    return above & (labels[faces[:, 0]] == labels[vertices[:, 2].argmax()])


def _neck_surface(body: AnnyBody, body_rest: torch.Tensor, seam: torch.Tensor) -> torch.Tensor:
    """``(F,)`` indices of Anny's triangles around the seam, all of them, cut or not."""
    centroids = body_rest[body.faces].mean(dim=1)
    return torch.nonzero((centroids[:, 2] - seam).abs() < NECK_SURFACE).squeeze(-1)


def _loop_at(faces: torch.Tensor, vertices: torch.Tensor, seam: torch.Tensor) -> torch.Tensor:
    """``(N,)`` the vertices of the mesh's boundary loop nearest the seam: where it was cut."""
    loops = boundary_loops(faces.cpu().numpy())
    distance = [(vertices[loop, 2] - seam).abs().mean().item() for loop in loops]
    return torch.from_numpy(loops[int(np.argmin(distance))]).to(vertices.device)


def _bend_neck(
    head: torch.Tensor,
    loop: torch.Tensor,
    body_rest: torch.Tensor,
    neck: torch.Tensor,
    blend: float,
) -> torch.Tensor:
    """The head's vertices with its cut neck moved onto the body's neck surface.

    The cut edge moves all the way; the neck above it follows less the farther up it is, and
    nothing moves more than ``blend`` metres from the edge.

    Args:
        neck: ``(F, 3)`` the triangles of the body's neck.
    """
    faces, weights = closest_points(head[loop], body_rest, neck)
    on_body = (weights.unsqueeze(-1) * body_rest[neck[faces]]).sum(dim=1)
    shift = on_body - head[loop]

    skin = head[ict.SKIN]
    distances, nearest = torch.cdist(skin, head[loop]).topk(4, largest=False)
    blend_weights = 1 / distances.clamp_min(1e-5)
    blend_weights = blend_weights / blend_weights.sum(dim=-1, keepdim=True)
    falloff = (1 - distances[:, 0] / blend).clamp(0, 1)
    falloff = falloff * falloff * (3 - 2 * falloff)
    moved = (blend_weights.unsqueeze(-1) * shift[nearest]).sum(dim=1) * falloff.unsqueeze(-1)
    head = head.clone()
    head[ict.SKIN] += moved
    logger.info("neck bent onto the body by up to %.1f mm", 1000 * shift.norm(dim=-1).max())
    return head


def _smooth_seam(vertices: torch.Tensor, faces: torch.Tensor, seam: torch.Tensor) -> torch.Tensor:
    """The vertices with the seam's zigzag smoothed out: triangles are cut whole, so the cut
    edges step up and down by a triangle's height."""
    laplacian = uniform_laplacian(faces, len(vertices))
    vertices = vertices.clone()
    for _ in range(SEAM_SMOOTHING):
        vertices[seam] -= 0.5 * (laplacian @ vertices)[seam]
    return vertices


def _surface_uvs(
    triangles: torch.Tensor, body: AnnyBody, body_rest: torch.Tensor, neck: torch.Tensor
) -> torch.Tensor:
    """``(T, 3, 2)`` body texture coordinates for ``(T, 3, 3)`` triangles lying on the body's
    neck: each triangle takes its corners' coordinates from the body triangle nearest its centre,
    so none spans a seam of the body's texture layout."""
    faces, _ = closest_points(triangles.mean(dim=1), body_rest, body.faces[neck])
    corners = body_rest[body.faces[neck[faces]]]  # (T, 3, 3)
    weights = plane_barycentric(triangles, *corners.unsqueeze(1).unbind(dim=2))
    return weights @ body.uvs[neck[faces]]


def match_heads(
    body: AnnyBody,
    body_rest: dict[str, torch.Tensor],
    head: IctFaceKit,
    head_rest: torch.Tensor,
) -> HeadPlacement:
    """Put the head model where Anny's head is: a first placement, refined against the video by
    ``fitting.placement``.

    Starts from Anny's nose and eye keypoints, then matches the surfaces of the two heads down
    to the upper neck: both are bald, so their shapes compare directly, and including the neck
    makes the head sit on the body's neck at the body's angle.
    """
    keypoints = body_rest["keypoints"]
    body_rest = body_rest["vertices"]
    landmarks = head_rest[head.landmarks] @ ICT_TO_WORLD.to(head_rest).T
    source = torch.stack((
        landmarks[ICT_NOSE_TIP],
        landmarks[list(ICT_LEFT_EYE)].mean(0),
        landmarks[list(ICT_RIGHT_EYE)].mean(0),
    ))  # fmt: skip
    target = keypoints[[COCO_NOSE, COCO_LEFT_EYE, COCO_RIGHT_EYE]]
    scale, rotation, translation = _umeyama(source, target)
    rotation = rotation @ ICT_TO_WORLD.to(rotation)

    lowest = target[1:, 2].mean() - MATCHED_BELOW_EYES
    body_head = torch.unique(body.faces[body.head_faces])
    body_skull = body_rest[body_head][body_rest[body_head, 2] > lowest]
    head_skin = head_rest[ict.SKIN]

    log_scale = torch.nn.Parameter(torch.tensor(scale, device=head_rest.device).log())
    rotvec = torch.nn.Parameter(roma.rotmat_to_rotvec(rotation))
    shift = torch.nn.Parameter(translation.clone())
    optimizer = torch.optim.Adam([log_scale, rotvec, shift], lr=1e-3)
    for _ in range(300):
        placed = log_scale.exp() * head_skin @ roma.rotvec_to_rotmat(rotvec).T + shift
        skull = placed[placed[:, 2] > lowest]
        distances = torch.cdist(skull, body_skull)
        loss = distances.min(dim=1).values.mean() + distances.min(dim=0).values.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    logger.info("heads matched: surface distance %.1f mm", 500 * loss.item())
    return HeadPlacement(
        log_scale.exp().item(), roma.rotvec_to_rotmat(rotvec).detach(), shift.detach()
    )


def transfer_weights(body: AnnyBody, body_rest: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """``(P, J)`` skinning weights for points, from their nearest body vertices.

    Eyeballs follow Anny's eye bones, so the runtime can turn them to look around.
    """
    distances, nearest = torch.cdist(points, body_rest).topk(4, largest=False)
    blend = 1 / distances.clamp_min(1e-4)
    weights = (body.skin_weights[nearest] * (blend / blend.sum(-1, keepdim=True))[..., None]).sum(1)
    for eyeball in (ict.LEFT_EYEBALL, ict.RIGHT_EYEBALL):
        bone = "eye.L" if points[eyeball, 0].mean() > 0 else "eye.R"  # Anny's left is +x
        weights[eyeball] = 0
        weights[eyeball, body.bone(bone)] = 1
    return weights


def _umeyama(
    source: torch.Tensor, target: torch.Tensor
) -> tuple[float, torch.Tensor, torch.Tensor]:
    """Scale, rotation and translation best mapping ``(P, 3)`` source points onto target points."""
    mu_s, mu_t = source.mean(0), target.mean(0)
    a, b = source - mu_s, target - mu_t
    u, s, vt = torch.linalg.svd(b.T @ a / len(a))
    d = torch.ones(3, device=a.device)
    d[2] = torch.sign(torch.linalg.det(u @ vt))
    rotation = u @ torch.diag(d) @ vt
    scale = float((s * d).sum() / a.square().sum(-1).mean())
    return scale, rotation, mu_t - scale * rotation @ mu_s
