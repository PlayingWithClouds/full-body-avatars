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
"""Anny's mesh parts of the head; they end near the chin."""
HAND_PARTS = ("hand.L", "hand.R")
FOOT_PARTS = ("foot.L", "foot.R")
BODY_CHANGES = (
    "l-upperarm", "r-upperarm", "l-lowerarm", "r-lowerarm",
    "breast", "nipple", "torso", "stomach", "pelvis", "buttocks", "hip-", "neck-",
    "l-upperleg", "r-upperleg", "l-lowerleg", "r-lowerleg", "l-leg", "r-leg",
    "upperlegs", "lowerlegs", "measure-",
)  # fmt: skip
"""Prefixes of the MakeHuman local changes fitted to the body: girth, volume and proportions of
the limbs and torso. The head's are left out (ICT-FaceKit replaces it), as are the hands' and
feet's, too small in the frames to fit."""
EXCLUDED_CHANGES = (
    "-trans-",
    "torso-scale-vert-incr", "measure-napetowaist-dist-incr", "measure-waisttohip-dist-incr",
    "neck-scale-vert-incr", "measure-neck-height-incr",
    "upperlegs-height-incr", "lowerlegs-height-incr", "nipple-point-incr",
)  # fmt: skip
"""Parts of labels of local changes not fitted: shifts of the torso, hips or neck as a whole
(the fit's translation and tilt place the body), changes that move the head (it is replaced at
the neck), duplicates of ``measure-*`` changes, and one that changes nothing."""


class AnnyBody(torch.nn.Module):
    """Anny driven by phenotype values and a rotation vector per bone.

    Poses are Anny's ``local-ref`` parameterization: each bone rotates about its own head, in
    world-aligned axes. Bone 0 is the root; its rotation and ``translation`` place the body.
    """

    def __init__(self):
        super().__init__()
        changes = [
            label
            for label in anny.Anny(local_changes="all").local_change_labels
            if label.startswith(BODY_CHANGES)
            and not any(part in label for part in EXCLUDED_CHANGES)
        ]
        self.model = anny.Anny(phenotypes="all", local_changes=changes).to(dtype=torch.float32)
        self.local_change_labels: tuple[str, ...] = tuple(self.model.local_change_labels)
        self.keypoint_regressor = anny.KeypointsRegressor.coco(
            self.model, labels=list(COCO_KEYPOINTS)
        )
        self.phenotype_labels: tuple[str, ...] = tuple(self.model.phenotype_labels)
        self.bone_labels: tuple[str, ...] = tuple(self.model.bone_labels)
        self.bone_parents: tuple[int, ...] = tuple(self.model.bone_parents)
        self.register_buffer("faces", self.model.faces.clone())
        self.register_buffer(
            "uvs",
            self.model.texture_coordinates[self.model.face_texture_coordinate_indices].float(),
        )
        self.register_buffer(
            "skin_weights",
            dense_bone_weights(
                self.model.vertex_bone_weights,
                self.model.vertex_bone_indices,
                self.model.bone_count,
            ).float(),
        )
        for name, parts in (("head", HEAD_PARTS), ("hand", HAND_PARTS), ("foot", FOOT_PARTS)):
            mask = get_face_segmentation_mask(self.model, list(parts))
            self.register_buffer(f"{name}_faces", mask)

    @property
    def num_bones(self) -> int:
        return len(self.bone_labels)

    def bone(self, label: str) -> int:
        return self.bone_labels.index(label)

    def default_phenotype(self, batch: int = 1) -> torch.Tensor:
        """``(batch, P)`` phenotype at the middle of every range."""
        return torch.full((batch, len(self.phenotype_labels)), 0.5, device=self.faces.device)

    def forward(
        self,
        phenotype: torch.Tensor,
        local_changes: torch.Tensor,
        rotations: torch.Tensor,
        translation: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Pose the body.

        Args:
            phenotype: ``(B, P)`` values in [0, 1], one per ``phenotype_labels``.
            local_changes: ``(B, L)`` values in [-1, 1], one per ``local_change_labels``; 0 is
                no change.
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
        output = self.model(
            pose_parameters=pose, phenotype_kwargs=phenotype, local_changes_kwargs=local_changes
        )
        output["keypoints"] = self.keypoint_regressor(output)
        output["skinning"] = output["bone_poses"] @ torch.linalg.inv(output["rest_bone_poses"])
        return output

    def skinning(
        self, rest_bone_poses: torch.Tensor, rotations: torch.Tensor, translation: torch.Tensor
    ) -> torch.Tensor:
        """``(B, J, 4, 4)`` the skinning matrices of ``forward``, for a shape already known.

        Fitting motion to one fixed shape needs only the bones: this skips rebuilding the rest
        mesh from the phenotype and skinning every vertex with every bone.

        Args:
            rest_bone_poses: ``(J, 4, 4)`` from ``forward``'s ``rest_bone_poses``.
            rotations: ``(B, J, 3)`` rotation vectors; ``translation``: ``(B, 3)``.
        """
        pose = torch.eye(4, device=rotations.device).repeat(*rotations.shape[:2], 1, 1)
        pose[..., :3, :3] = roma.rotvec_to_rotmat(rotations)
        pose[:, 0, :3, 3] = translation
        rest = rest_bone_poses.expand(len(rotations), -1, -1, -1)
        _, bone_poses = self.model.get_bone_transforms(pose, rest)
        return bone_poses @ torch.linalg.inv(rest_bone_poses)
