"""Gaussians bound to the triangles of the avatar's mesh, and their rendering."""

from avatar_splats.gaussians.bound import BoundGaussians, GaussianRecord
from avatar_splats.gaussians.render import render

__all__ = ["BoundGaussians", "GaussianRecord", "render"]
