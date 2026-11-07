"""Gaussians that ride on a deforming triangle mesh.

Each Gaussian belongs to one triangle. It sits at a fixed barycentric point of the triangle, moved
by an offset expressed in the triangle's frame (along its first edge, across it, and along its
normal); its rotation is relative to that frame. Offsets and scales are in units of the triangle's
size, so a Gaussian keeps its place and proportions however the mesh is posed, stretched or scaled.
This is the computation the game runtime repeats every frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import roma
import torch

from avatar_splats.body_models.mesh import triangle_frames
from avatar_splats.records import ArrayRecord

SH_C0 = 0.28209479177387814
"""Value of the constant spherical harmonic; colours are 0.5 plus SH_C0 times its coefficient."""


@dataclass(frozen=True)
class GaussianRecord(ArrayRecord):
    """Trained Gaussians as arrays; see ``BoundGaussians`` for their meaning."""

    faces: np.ndarray
    barycentric: np.ndarray
    offsets: np.ndarray
    rotations: np.ndarray
    log_scales: np.ndarray
    opacity_logits: np.ndarray
    sh: np.ndarray


class BoundGaussians(torch.nn.Module):
    faces: torch.Tensor
    """``(G,)`` the triangle each Gaussian is bound to."""
    barycentric: torch.Tensor
    """``(G, 3)`` where on its triangle the Gaussian is anchored."""

    def __init__(self, record: GaussianRecord):
        super().__init__()
        self.register_buffer("faces", torch.from_numpy(record.faces).long())
        self.register_buffer("barycentric", torch.from_numpy(record.barycentric).float())
        self.offsets = torch.nn.Parameter(torch.from_numpy(record.offsets).float())
        """``(G, 3)`` offset from the anchor in the triangle's frame, in triangle sizes."""
        self.rotations = torch.nn.Parameter(torch.from_numpy(record.rotations).float())
        """``(G, 4)`` rotation relative to the triangle's frame, as a wxyz quaternion."""
        self.log_scales = torch.nn.Parameter(torch.from_numpy(record.log_scales).float())
        """``(G, 3)`` log of the extent along the Gaussian's axes, in triangle sizes."""
        self.opacity_logits = torch.nn.Parameter(torch.from_numpy(record.opacity_logits).float())
        sh = torch.from_numpy(record.sh).float()
        self.sh_dc = torch.nn.Parameter(sh[:, :1].contiguous())
        """``(G, 1, 3)`` the view-independent colour coefficient."""
        self.sh_rest = torch.nn.Parameter(sh[:, 1:].contiguous())
        """``(G, K - 1, 3)`` view-dependent colour coefficients."""

    @classmethod
    def initialize(
        cls,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        density: torch.Tensor,
        count: int,
        sh_degree: int,
        seed: int = 0,
    ) -> BoundGaussians:
        """Spread ``count`` small grey Gaussians over a mesh.

        Args:
            vertices: ``(V, 3)`` rest-pose vertices.
            faces: ``(F, 3)`` triangles.
            density: ``(F,)`` relative number of Gaussians per area on each triangle.
        """
        frames, size = triangle_frames(vertices, faces)
        area = size.square() / 2
        weight = area * density
        generator = torch.Generator(device=vertices.device).manual_seed(seed)
        chosen = torch.multinomial(weight, count, replacement=True, generator=generator)
        u, v = torch.rand(2, count, device=vertices.device, generator=generator)
        flip = u + v > 1
        u, v = torch.where(flip, 1 - u, u), torch.where(flip, 1 - v, v)

        # Each Gaussian covers about the area its triangle offers it, flattened along the normal.
        spacing = (weight.sum() / (count * density[chosen])).sqrt() / size[chosen]
        log_scales = spacing.log().unsqueeze(-1).repeat(1, 3)
        log_scales[:, 2] += math.log(0.3)
        rotations = torch.zeros(count, 4, device=vertices.device)
        rotations[:, 0] = 1
        return cls(
            GaussianRecord(
                faces=chosen.cpu().numpy(),
                barycentric=torch.stack((1 - u - v, u, v), dim=-1).cpu().numpy(),
                offsets=np.zeros((count, 3), np.float32),
                rotations=rotations.cpu().numpy(),
                log_scales=log_scales.cpu().numpy(),
                opacity_logits=np.zeros(count, np.float32),
                sh=np.zeros((count, (sh_degree + 1) ** 2, 3), np.float32),
            )
        )

    def record(self) -> GaussianRecord:
        def array(tensor: torch.Tensor) -> np.ndarray:
            return tensor.detach().cpu().numpy()

        return GaussianRecord(
            faces=array(self.faces),
            barycentric=array(self.barycentric),
            offsets=array(self.offsets),
            rotations=array(self.rotations),
            log_scales=array(self.log_scales),
            opacity_logits=array(self.opacity_logits),
            sh=array(torch.cat((self.sh_dc, self.sh_rest), dim=1)),
        )

    def __len__(self) -> int:
        return len(self.faces)

    def sh(self) -> torch.Tensor:
        return torch.cat((self.sh_dc, self.sh_rest), dim=1)

    def opacities(self) -> torch.Tensor:
        return torch.sigmoid(self.opacity_logits)

    def place(
        self, vertices: torch.Tensor, faces: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """World means, wxyz rotations and scales of the Gaussians on ``(V, 3)`` posed vertices.

        Returns:
            ``(G, 3)`` means, ``(G, 4)`` rotations and ``(G, 3)`` scales.
        """
        frames, size = triangle_frames(vertices, faces[self.faces])
        corners = vertices[faces[self.faces]]
        anchors = (self.barycentric.unsqueeze(-1) * corners).sum(dim=1)
        size = size.unsqueeze(-1)
        means = anchors + (frames @ (self.offsets * size).unsqueeze(-1)).squeeze(-1)
        local = roma.unitquat_to_rotmat(
            _wxyz_to_xyzw(torch.nn.functional.normalize(self.rotations, dim=-1))
        )
        rotations = _xyzw_to_wxyz(roma.rotmat_to_unitquat(frames @ local))
        return means, rotations, self.log_scales.exp() * size


def _wxyz_to_xyzw(q: torch.Tensor) -> torch.Tensor:
    return q[..., [1, 2, 3, 0]]


def _xyzw_to_wxyz(q: torch.Tensor) -> torch.Tensor:
    return q[..., [3, 0, 1, 2]]
