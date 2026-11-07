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
