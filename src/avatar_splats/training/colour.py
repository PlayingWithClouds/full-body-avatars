"""Matching the colours of the videos.

The videos were generated separately and are not graded alike: the same skin comes out a little
warmer or brighter in one than in another, which shows as a step in tone where the head meets
the body. The avatar keeps one set of colours, the body video's, and learns for each other video
the affine colour change that turns them into that video's.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from avatar_splats.records import ArrayRecord


@dataclass(frozen=True)
class ColourRecord(ArrayRecord):
    reference: str
    videos: np.ndarray
    """``(N,)`` the videos other than the reference."""
    matrices: np.ndarray
    """``(N, 3, 3)`` linear colour change of each video."""
    offsets: np.ndarray
    """``(N, 3)`` colour offset of each video."""


class ColourCorrection(torch.nn.Module):
    def __init__(self, videos: list[str], reference: str):
        super().__init__()
        self.reference = reference
        self.matrices = torch.nn.ParameterDict(
            {video: torch.nn.Parameter(torch.eye(3)) for video in videos if video != reference}
        )
        self.offsets = torch.nn.ParameterDict(
            {video: torch.nn.Parameter(torch.zeros(3)) for video in videos if video != reference}
        )

    def forward(self, video: str, colour: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """``(H, W, 3)`` the avatar's colour as ``video`` shows it.

        Args:
            colour: ``(H, W, 3)`` the avatar rendered over black, so premultiplied by alpha.
            alpha: ``(H, W)`` its opacity.
        """
        if video == self.reference:
            return colour
        return colour @ self.matrices[video].T + alpha.unsqueeze(-1) * self.offsets[video]

    def to_reference(self, video: str, colour: torch.Tensor) -> torch.Tensor:
        """``(..., 3)`` colours as ``video`` shows them, turned into the reference video's."""
        if video == self.reference:
            return colour
        matrix, offset = self.matrices[video].detach(), self.offsets[video].detach()
        return (colour - offset) @ torch.linalg.inv(matrix).T

    def record(self) -> ColourRecord:
        videos = list(self.matrices)
        return ColourRecord(
            reference=self.reference,
            videos=np.array(videos),
            matrices=np.stack([self.matrices[v].detach().cpu().numpy() for v in videos]),
            offsets=np.stack([self.offsets[v].detach().cpu().numpy() for v in videos]),
        )

    @classmethod
    def from_record(cls, record: ColourRecord) -> ColourCorrection:
        videos = record.videos.tolist()
        colours = cls([record.reference, *videos], record.reference)
        with torch.no_grad():
            for video, matrix, offset in zip(videos, record.matrices, record.offsets, strict=True):
                colours.matrices[video].copy_(torch.from_numpy(matrix))
                colours.offsets[video].copy_(torch.from_numpy(offset))
        return colours
