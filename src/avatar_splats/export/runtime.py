"""A reference runtime for ``.avatar`` files: what a game engine does to pose and draw the avatar.

Per frame:

1. Forward kinematics: the animation gives each bone a rotation relative to its bind pose, and
   the root an offset from its bind position (root motion); the bones' world transforms follow
   down the hierarchy (``forward_kinematics``).
2. Skinning matrices: each bone's world transform times the inverse of its bind pose.
3. Expressions: the head's rest vertices plus the expression blendshapes, weighted.
4. Linear blend skinning of all vertices.
5. The mesh is drawn with its two textures, filling the depth buffer.
6. Each Gaussian is placed on its triangle (``BoundGaussians.place``): anchor point, offset,
   rotation and scale in the triangle's frame.
7. The Gaussians are splatted over the mesh, colours from their spherical harmonics plus 0.5,
   each depth-tested against the mesh with a margin of ``occlusion_margin`` triangle sizes.

Several avatars in one scene are drawn together: their meshes into one depth buffer, then all
their Gaussians in one pass, sorted together, so each hides behind the others where it should.
"""

from __future__ import annotations

import numpy as np
import torch

from avatar_splats.body_models.hybrid import Avatar, Rig
from avatar_splats.cameras import Camera
from avatar_splats.export.avatar_file import AvatarFile
from avatar_splats.gaussians import BoundGaussians, GaussianRecord
from avatar_splats.rendering import composite, render_mesh


def forward_kinematics(
    bind_poses: torch.Tensor,
    parents: list[int],
    rotations: torch.Tensor,
    root_offset: torch.Tensor | None = None,
) -> torch.Tensor:
    """World transforms of bones rotated relative to their bind pose.

    Args:
        bind_poses: ``(J, 4, 4)`` bone frames in the bind pose; parents come before children.
        parents: parent index per bone, -1 for the root.
        rotations: ``(B, J, 3, 3)`` rotation of each bone in its own bind frame.
        root_offset: ``(B, 3)`` the root's offset from its bind position, in world axes.

    Returns:
        ``(B, J, 4, 4)`` bone transforms in the world.
    """
    local = rotations.new_zeros(*rotations.shape[:-2], 4, 4)
    local[..., :3, :3] = rotations
    local[..., 3, 3] = 1
    world: list[torch.Tensor] = []
    for bone, parent in enumerate(parents):
        relative = (
            bind_poses[bone]
            if parent < 0
            else torch.linalg.inv(bind_poses[parent]) @ bind_poses[bone]
        )
        base = relative if parent < 0 else world[parent] @ relative
        transform = base @ local[:, bone]
        if parent < 0 and root_offset is not None:
            transform = transform.clone()
            transform[:, :3, 3] += root_offset
        world.append(transform)
    return torch.stack(world, dim=1)


def bind_relative(bind_poses: torch.Tensor, rotations: torch.Tensor) -> torch.Tensor:
    """Express ``(..., J, 3, 3)`` rotations given in world axes in each bone's bind frame.

    Handy for posing by hand: "turn the left upper arm 60 degrees about world x".
    """
    frames = bind_poses[:, :3, :3]
    return frames.transpose(-1, -2) @ rotations @ frames


class AvatarRuntime(torch.nn.Module):
    """Poses an ``.avatar`` file's mesh and places its Gaussians."""

    def __init__(self, file: AvatarFile):
        super().__init__()
        arrays, metadata = file.arrays, file.metadata
        self.bone_names: list[str] = metadata["bones"]
        self.expression_names: list[str] = metadata["expressions"]
        self.sh_degree: int = metadata["sh_degree"]
        self.parents = arrays["bone_parents"].tolist()
        self.rig = Rig(
            Avatar(
                rest_vertices=arrays["rest_vertices"],
                faces=arrays["faces"].astype(np.int64),
                face_parts=arrays["face_parts"].astype(np.int8),
                uvs=arrays["uvs"],
                face_textures=arrays["face_textures"].astype(np.int8),
                bone_indices=arrays["bone_indices"].astype(np.int64),
                bone_weights=arrays["bone_weights"],
                bone_labels=np.array(self.bone_names),
                bone_parents=arrays["bone_parents"],
                bind_poses=arrays["bind_poses"],
                head_start=metadata["head_start"],
                expression_names=np.array(self.expression_names),
                expression_basis=arrays["expression_basis"],
            )
        )
        self.occlusion_margin: float = metadata["occlusion_margin"]
        self.register_buffer("uvs", torch.from_numpy(arrays["uvs"]))
        self.register_buffer("face_textures", torch.from_numpy(arrays["face_textures"]).long())
        for name in metadata["textures"]:
            self.register_buffer(name, torch.from_numpy(arrays[name]).float() / 255)
        self.texture_names: list[str] = metadata["textures"]
        opacities = arrays["gaussian_opacities"].clip(1e-7, 1 - 1e-7)
        self.gaussians = BoundGaussians(
            GaussianRecord(
                faces=arrays["gaussian_faces"].astype(np.int64),
                barycentric=arrays["gaussian_barycentric"],
                offsets=arrays["gaussian_offsets"],
                rotations=arrays["gaussian_rotations"],
                log_scales=arrays["gaussian_log_scales"],
                opacity_logits=np.log(opacities / (1 - opacities)),
                sh=arrays["gaussian_sh"],
            )
        )

    def skinning(
        self, rotations: torch.Tensor, root_offset: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``(B, J, 4, 4)`` skinning matrices for ``(B, J, 3, 3)`` bind-relative bone rotations
        and ``(B, 3)`` root offsets."""
        world = forward_kinematics(self.rig.bind_poses, self.parents, rotations, root_offset)
        return world @ torch.linalg.inv(self.rig.bind_poses)

    def pose(
        self,
        rotations: torch.Tensor,
        expressions: torch.Tensor | None = None,
        root_offset: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``(B, V, 3)`` vertices for bone rotations, ``(B, E)`` expression weights and ``(B, 3)``
        root offsets."""
        return self.rig(self.skinning(rotations, root_offset), expressions)

    def place(self, vertices: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Means, wxyz rotations, scales, opacities and SH of the Gaussians on ``(V, 3)``."""
        means, rotations, scales = self.gaussians.place(vertices, self.rig.faces)
        return means, rotations, scales, self.gaussians.opacities(), self.gaussians.sh()

    def render(
        self, vertices: torch.Tensor, camera: Camera, background: torch.Tensor
    ) -> torch.Tensor:
        """``(H, W, 3)`` image of the avatar posed as ``(V, 3)`` vertices, or of several copies
        of it posed as ``(P, V, 3)``, over a ``(3,)`` background colour."""
        copies = vertices.reshape(-1, *vertices.shape[-2:])
        faces = self.rig.faces
        count = copies.shape[1]
        all_faces = torch.cat([faces + i * count for i in range(len(copies))])
        mesh = render_mesh(
            copies.flatten(0, 1),
            all_faces,
            self.uvs.repeat(len(copies), 1, 1),
            self.face_textures.repeat(len(copies)),
            [getattr(self, name) for name in self.texture_names],
            camera,
            torch.zeros_like(background),
        )
        placed = [self.place(v) for v in copies]
        means, rotations, scales, opacities, sh = (
            torch.cat(parts) for parts in zip(*placed, strict=True)
        )
        margins = self.occlusion_margin * torch.cat(
            [self.gaussians.triangle_sizes(v, faces) for v in copies]
        )
        image, alpha = composite(
            means, rotations, scales, opacities, sh, margins, mesh, camera, self.sh_degree
        )
        return image + (1 - alpha).unsqueeze(-1) * background
