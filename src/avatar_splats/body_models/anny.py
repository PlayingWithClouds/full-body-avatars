"""The Anny body model, with the conventions the rest of the pipeline uses.

Anny (NAVER LABS; code Apache-2.0, MakeHuman assets CC0) works in metres with z up; the body faces
-y and its left is +x. Only the default ``anny`` topology and rig are used: the ``smpl`` and
``smplx`` topologies load non-commercial data.
"""

from __future__ import annotations

import anny
import roma
import torch
from anny.face_segmentation import get_face_segmentation_mask
from anny.skinning.skinning import dense_bone_weights

from avatar_splats.preprocess.keypoints import COCO_KEYPOINTS

HEAD_PARTS = (
    "head",
    "eye_cavity.L",
    "eye_cavity.R",
    "eye_front.L",
    "eye_back.L",
    "eye_front.R",
    "eye_back.R",
    "mouth_cavity",
    "tongue",
)
"""Anny's mesh parts above the shoulders; the neck belongs to "head"."""


class AnnyBody(torch.nn.Module):
    """Anny driven by phenotype values and a rotation vector per bone.

    Poses are Anny's ``local-ref`` parameterization: each bone rotates about its own head, in
    world-aligned axes. Bone 0 is the root; its rotation and ``translation`` place the body.
    """

    def __init__(self):
        super().__init__()
        self.model = anny.Anny(phenotypes="all").to(dtype=torch.float32)
        self.keypoint_regressor = anny.KeypointsRegressor.coco(
            self.model, labels=list(COCO_KEYPOINTS)
        )
        self.phenotype_labels: tuple[str, ...] = tuple(self.model.phenotype_labels)
        self.bone_labels: tuple[str, ...] = tuple(self.model.bone_labels)
        self.bone_parents: tuple[int, ...] = tuple(self.model.bone_parents)
        self.register_buffer("faces", self.model.faces.clone())
        self.register_buffer(
            "skin_weights",
            dense_bone_weights(
                self.model.vertex_bone_weights,
                self.model.vertex_bone_indices,
                self.model.bone_count,
            ).float(),
        )
        self.register_buffer("head_faces", get_face_segmentation_mask(self.model, list(HEAD_PARTS)))

    @property
    def num_bones(self) -> int:
        return len(self.bone_labels)

    def bone(self, label: str) -> int:
        return self.bone_labels.index(label)

    def default_phenotype(self, batch: int = 1) -> torch.Tensor:
        """``(batch, P)`` phenotype at the middle of every range."""
        return torch.full((batch, len(self.phenotype_labels)), 0.5, device=self.faces.device)

    def forward(
        self, phenotype: torch.Tensor, rotations: torch.Tensor, translation: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Pose the body.

        Args:
            phenotype: ``(B, P)`` values in [0, 1], one per ``phenotype_labels``.
            rotations: ``(B, J, 3)`` rotation vectors, one per bone.
            translation: ``(B, 3)`` root translation in metres.

        Returns:
            ``vertices`` ``(B, V, 3)``, ``rest_vertices`` ``(B, V, 3)``, ``keypoints``
            ``(B, 17, 3)`` COCO keypoints, ``bone_poses`` ``(B, J, 4, 4)`` posed bone frames,
            ``rest_bone_poses`` ``(B, J, 4, 4)`` and ``skinning`` ``(B, J, 4, 4)``, the matrices
            taking rest-pose vertices to posed ones.
        """
        pose = torch.eye(4, device=rotations.device).repeat(*rotations.shape[:2], 1, 1)
        pose[..., :3, :3] = roma.rotvec_to_rotmat(rotations)
        pose[:, 0, :3, 3] = translation
        output = self.model(pose_parameters=pose, phenotype_kwargs=phenotype)
        output["keypoints"] = self.keypoint_regressor(output)
        output["skinning"] = output["bone_poses"] @ torch.linalg.inv(output["rest_bone_poses"])
        return output
