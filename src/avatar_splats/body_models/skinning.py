"""Linear blend skinning: the deformation the game runtime applies to the avatar."""

from __future__ import annotations

import torch


def linear_blend_skinning(
    rest_vertices: torch.Tensor,
    bone_indices: torch.Tensor,
    bone_weights: torch.Tensor,
    transforms: torch.Tensor,
) -> torch.Tensor:
    """Move rest-pose vertices by the weighted transforms of the bones they are bound to.

    Args:
        rest_vertices: ``(..., V, 3)`` vertices in the rest pose.
        bone_indices: ``(V, K)`` the bones each vertex is bound to.
        bone_weights: ``(V, K)`` weights of those bones, summing to 1 per vertex.
        transforms: ``(..., J, 4, 4)`` skinning matrices, mapping rest to posed space per bone.

    Returns:
        ``(..., V, 3)`` posed vertices.
    """
    blended = torch.einsum("vk,...vkij->...vij", bone_weights, transforms[..., bone_indices, :, :])
    return (blended[..., :3, :3] @ rest_vertices.unsqueeze(-1)).squeeze(-1) + blended[..., :3, 3]


def sparsify_weights(dense: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep the ``k`` largest of ``(V, J)`` skinning weights per vertex, renormalized.

    Returns:
        ``(V, k)`` bone indices and weights.
    """
    weights, indices = dense.topk(k, dim=-1)
    return indices, weights / weights.sum(dim=-1, keepdim=True)


def skinned_point_weights(
    regressor: torch.Tensor,
    rest_vertices: torch.Tensor,
    bone_indices: torch.Tensor,
    bone_weights: torch.Tensor,
    num_bones: int,
) -> torch.Tensor:
    """Weights that give points regressed from skinned vertices straight from the skinning
    matrices, without skinning the vertices.

    A point that is a fixed weighted sum of vertices, each moved by linear blend skinning, is a
    weighted sum of the bones' skinning matrices applied to fixed rest-pose points. With the
    returned ``(K, J, 4)`` weights ``Y``, the points are ``(S[..., :3, :] * Y).sum`` over bones
    and columns (``skinned_points``): exactly ``regressor @ linear_blend_skinning(...)``.

    Args:
        regressor: ``(K, V)`` each point's weights over the vertices.
        rest_vertices: ``(V, 3)``; ``bone_indices``, ``bone_weights``: ``(V, I)`` as in
            ``linear_blend_skinning``.
    """
    dense = torch.zeros(len(rest_vertices), num_bones, device=rest_vertices.device)
    dense.scatter_add_(1, bone_indices, bone_weights)
    homogeneous = torch.nn.functional.pad(rest_vertices, (0, 1), value=1.0)
    per_bone = (dense.unsqueeze(-1) * homogeneous.unsqueeze(1)).flatten(1)  # (V, J * 4)
    return (regressor @ per_bone).view(len(regressor), num_bones, 4)


def skinned_points(skinning: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """``(..., K, 3)`` points for ``(..., J, 4, 4)`` skinning matrices and ``(K, J, 4)``
    weights from ``skinned_point_weights``."""
    return torch.einsum("...jab,kjb->...ka", skinning[..., :3, :], weights)
