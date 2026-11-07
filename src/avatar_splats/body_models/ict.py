"""The ICT-FaceKit light face model: identity and expression blendshapes on one head mesh.

https://github.com/USC-ICT/ICT-FaceKit (MIT). The model ships as OBJ files: a neutral mesh and, for
every identity mode and expression, a full mesh whose difference from the neutral one is that
blendshape. Identity weights are standard normal; expression weights run from 0 to 1.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from avatar_splats.body_models.mesh import read_obj, read_obj_uvs, read_obj_vertices

CENTIMETRE = 0.01
"""ICT-FaceKit is modelled in centimetres; this module works in metres."""

NUM_KEPT_VERTICES = 24591
NUM_KEPT_FACES = 24692
"""The model's regions are stored in order. The first ones are surfaces to render: face, head and
neck, mouth socket, eye sockets, gums and tongue, teeth, eyeballs. The rest (lacrimal fluid, eye
blend and occlusion meshes, eyelash strips) are shader helpers for Unreal and are dropped."""

SKIN = slice(0, 11248)
"""Face, head and neck: the outer surface."""
FACE = slice(0, 9409)
MOUTH_INTERIOR = slice(11248, 21451)
"""Mouth socket, eye sockets, gums, tongue and teeth."""
EYEBALLS = slice(21451, 24591)
LEFT_EYEBALL = slice(21451, 23021)
RIGHT_EYEBALL = slice(23021, 24591)

UV_TILES = 5
"""Tiles side by side in the head's texture atlas. ICT-FaceKit lays its UVs out in four tiles
(u from 0 to 4: the face; the head, neck and mouth; the mouth's interior); the eyeballs' UVs
overlap the face's, so they move to a fifth tile."""


class IctFaceKit(torch.nn.Module):
    """Head vertices as the neutral mesh plus weighted identity and expression blendshapes.

    Coordinates are in metres, y up, the face looking along +z.
    """

    neutral: torch.Tensor
    """``(V, 3)`` neutral vertices."""
    faces: torch.Tensor
    """``(F, 3)`` triangles."""
    identity_basis: torch.Tensor
    """``(I, V, 3)`` identity blendshapes for weights of standard deviation 1."""
    expression_basis: torch.Tensor
    """``(E, V, 3)`` expression blendshapes for weights from 0 to 1."""
    landmarks: torch.Tensor
    """``(68,)`` vertex indices of the Multi-PIE 68 facial landmarks."""
    uvs: torch.Tensor
    """``(F, 3, 2)`` texture coordinates of the triangles' corners in the head's atlas, in
    [0, 1] with v up."""

    def __init__(self, directory: str | Path, num_identities: int = 100):
        super().__init__()
        model = Path(directory) / "FaceXModel"
        config = json.loads((model / "vertex_indices.json").read_text())

        neutral, faces = read_obj(model / "generic_neutral_mesh.obj")
        neutral = neutral[:NUM_KEPT_VERTICES]
        kept = faces.max(axis=1) < NUM_KEPT_VERTICES
        faces = faces[kept]
        uvs = read_obj_uvs(model / "generic_neutral_mesh.obj")[kept]
        uvs[faces.min(axis=1) >= EYEBALLS.start, :, 0] += UV_TILES - 1
        uvs[..., 0] /= UV_TILES

        def basis(names: list[str]) -> torch.Tensor:
            meshes = [
                read_obj_vertices(model / f"{name}.obj")[:NUM_KEPT_VERTICES] for name in names
            ]
            return torch.from_numpy((np.stack(meshes) - neutral) * CENTIMETRE)

        self.expression_names: tuple[str, ...] = tuple(config["expressions"])
        self.register_buffer("neutral", torch.from_numpy(neutral * CENTIMETRE))
        self.register_buffer("faces", torch.from_numpy(faces))
        self.register_buffer("uvs", torch.from_numpy(uvs))
        self.register_buffer(
            "identity_basis", basis([f"identity{i:03d}" for i in range(num_identities)])
        )
        self.register_buffer("expression_basis", basis(self.expression_names))
        self.register_buffer("landmarks", torch.tensor(config["idx_to_landmark_verts"]))

    @property
    def num_identities(self) -> int:
        return len(self.identity_basis)

    @property
    def num_expressions(self) -> int:
        return len(self.expression_basis)

    def forward(self, identity: torch.Tensor, expression: torch.Tensor) -> torch.Tensor:
        """``(..., V, 3)`` vertices for ``(..., I)`` identity and ``(..., E)`` expressions."""
        return (
            self.neutral
            + torch.einsum("...i,ivc->...vc", identity, self.identity_basis)
            + torch.einsum("...e,evc->...vc", expression, self.expression_basis)
        )
