"""The avatar's mesh: the fitted Anny body with its head replaced by the fitted ICT-FaceKit head.

The head is placed on the body by aligning the two heads. Both meshes are then cut at a seam
``neck_seam`` below the chin: the head model keeps the head and upper neck (its lower neck flares
out into a collar), Anny keeps the rest, and the two overlap by ``neck_overlap`` around the seam.
Gaussians, not the mesh, are what gets rendered, so the overlap needs no stitching: the head's
vertices take their skinning weights from Anny's nearest vertices, and the two necks move together.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import roma
import torch

from avatar_splats.body_models import ict
from avatar_splats.body_models.anny import AnnyBody
from avatar_splats.body_models.ict import IctFaceKit
from avatar_splats.body_models.skinning import linear_blend_skinning, sparsify_weights
from avatar_splats.config import BodyModelsConfig
from avatar_splats.records import ArrayRecord

logger = logging.getLogger(__name__)

BODY, HEAD_SKIN, MOUTH, EYES = 0, 1, 2, 3
"""Parts of the avatar's mesh, as stored in ``Avatar.face_parts``."""

ICT_TO_WORLD = torch.tensor([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])
"""ICT-FaceKit's axes (y up, facing +z) to Anny's (z up, facing -y)."""

COCO_NOSE, COCO_LEFT_EYE, COCO_RIGHT_EYE = 0, 1, 2
ICT_NOSE_TIP, ICT_RIGHT_EYE, ICT_LEFT_EYE = 30, (36, 39), (42, 45)
"""Multi-PIE landmarks matching Anny's COCO nose and eye keypoints."""
ICT_CHIN = 8
MATCHED_BELOW_EYES = 0.14
"""How far below the eyes, in metres, the head surfaces are matched to place the head."""


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
    """``(F,)`` each triangle's part: ``BODY``, ``HEAD_SKIN``, ``MOUTH`` or ``EYES``."""
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
    """Index of the first head vertex; the head's vertices run to the end."""
    expression_names: np.ndarray
    """``(E,)`` ARKit-style expression names."""
    expression_basis: np.ndarray
    """``(E, V - head_start, 3)`` head vertex offsets of each expression at weight 1."""


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
        self, skinning: torch.Tensor, expressions: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``(B, V, 3)`` vertices for ``(B, J, 4, 4)`` skinning and ``(B, E)`` expressions."""
        rest = self.rest_vertices.expand(len(skinning), -1, -1)
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
    body: AnnyBody, phenotype: np.ndarray, offsets: np.ndarray
) -> dict[str, torch.Tensor]:
    """The fitted body in its rest pose: ``vertices`` ``(V, 3)``, ``keypoints`` ``(17, 3)`` and
    ``bind_poses`` ``(J, 4, 4)``."""
    device = body.faces.device
    out = body(
        torch.from_numpy(phenotype)[None].to(device),
        torch.zeros(1, body.num_bones, 3, device=device),
        torch.zeros(1, 3, device=device),
    )
    vertices = out["rest_vertices"][0] + torch.from_numpy(offsets).to(device)
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
    """Join the fitted body and head, placed on it, into one rigged mesh."""
    device = body.faces.device
    bind_poses = body_rest["bind_poses"]
    body_rest = body_rest["vertices"]
    head = head.to(device)
    scale, rotation = placement.scale, placement.rotation
    head_world = placement.apply(head_rest)

    seam = head_world[head.landmarks[ICT_CHIN], 2] - config.neck_seam
    centroids = body_rest[body.faces].mean(dim=1)
    removed = body.head_faces & (centroids[:, 2] > seam + config.neck_overlap / 2)
    body_faces = body.faces[~removed]
    head_faces = head.faces
    head_centroids = head_world[head_faces].mean(dim=1)
    neck = (head_faces.min(dim=1).values >= ict.FACE.stop) & (
        head_faces.max(dim=1).values < ict.SKIN.stop
    )
    head_faces = head_faces[~(neck & (head_centroids[:, 2] < seam - config.neck_overlap / 2))]
    kept = torch.unique(body_faces)
    remap = torch.full((len(body_rest),), -1, dtype=torch.long, device=device)
    remap[kept] = torch.arange(len(kept), device=device)

    k = config.max_bone_influences
    body_bones, body_weights = sparsify_weights(body.skin_weights[kept], k)
    head_bones, head_weights = sparsify_weights(transfer_weights(body, body_rest, head_world), k)

    parts = torch.full((len(head_faces),), HEAD_SKIN, device=device)
    parts[head_faces.min(dim=1).values >= ict.MOUTH_INTERIOR.start] = MOUTH
    parts[head_faces.min(dim=1).values >= ict.EYEBALLS.start] = EYES

    expression_basis = scale * head.expression_basis @ rotation.T
    logger.info(
        "avatar: %d body and %d head vertices, head scaled by %.3f",
        len(kept),
        len(head_world),
        scale,
    )
    return Avatar(
        rest_vertices=torch.cat((body_rest[kept], head_world)).cpu().numpy(),
        faces=torch.cat((remap[body_faces], head_faces + len(kept))).cpu().numpy(),
        face_parts=torch.cat((torch.full((len(body_faces),), BODY, device=device), parts))
        .to(torch.int8)
        .cpu()
        .numpy(),
        bone_indices=torch.cat((body_bones, head_bones)).cpu().numpy(),
        bone_weights=torch.cat((body_weights, head_weights)).cpu().numpy(),
        bone_labels=np.array(body.bone_labels),
        bone_parents=np.array(body.bone_parents),
        bind_poses=bind_poses.cpu().numpy(),
        head_start=len(kept),
        expression_names=np.array(head.expression_names),
        expression_basis=expression_basis.cpu().numpy(),
    )


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
