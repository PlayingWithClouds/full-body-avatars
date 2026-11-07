"""Pinhole cameras."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def intrinsics(width: int, height: int, fov_y_degrees: float) -> torch.Tensor:
    """``(3, 3)`` intrinsics of a centred pinhole camera with the given vertical field of view."""
    focal = 0.5 * height / math.tan(math.radians(fov_y_degrees) / 2)
    return torch.tensor([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])


def world_to_camera() -> torch.Tensor:
    """``(4, 4)`` view matrix of the fixed camera: at the origin, looking along world +y, z up."""
    view = torch.eye(4)
    view[:3, :3] = torch.tensor([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])
    return view


@dataclass(frozen=True)
class Camera:
    """A camera and the size of the images it takes."""

    K: torch.Tensor
    """``(3, 3)`` intrinsics in pixels."""
    view: torch.Tensor
    """``(4, 4)`` world-to-camera transform."""
    width: int
    height: int

    @classmethod
    def fixed(cls, width: int, height: int, fov_y_degrees: float) -> Camera:
        return cls(intrinsics(width, height, fov_y_degrees), world_to_camera(), width, height)

    def scaled(self, factor: float) -> Camera:
        """The same camera taking images ``factor`` times as large on each side."""
        K = self.K.clone()
        K[:2] *= factor
        return Camera(K, self.view, round(self.width * factor), round(self.height * factor))

    def to(self, device: torch.device | str) -> Camera:
        return Camera(self.K.to(device), self.view.to(device), self.width, self.height)

    def project(self, points: torch.Tensor) -> torch.Tensor:
        """Project ``(..., 3)`` world points to ``(..., 2)`` pixel coordinates."""
        camera = points @ self.view[:3, :3].T + self.view[:3, 3]
        pixels = camera @ self.K.T
        return pixels[..., :2] / pixels[..., 2:].clamp_min(1e-6)

    def depth_for_height(self, world_height: float, pixel_height: float) -> float:
        """Distance at which something ``world_height`` tall spans ``pixel_height`` pixels."""
        return float(self.K[1, 1]) * world_height / pixel_height

    def unproject(self, pixel: torch.Tensor, depth: float) -> torch.Tensor:
        """The world point at ``depth`` along the ray through a ``(2,)`` pixel."""
        ray = torch.linalg.solve(self.K, torch.cat((pixel, pixel.new_ones(1))))
        camera = ray * depth / ray[2]
        return self.view[:3, :3].T @ (camera - self.view[:3, 3])
