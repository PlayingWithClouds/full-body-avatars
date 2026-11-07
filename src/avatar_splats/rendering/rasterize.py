"""Rasterizing triangles in PyTorch: which triangle each pixel sees, and where on it.

Which triangle is nearest at each pixel is found without gradients: every pixel inside a
triangle's bounding box is tested, and the nearest depth wins through one ``scatter_reduce``.
The barycentric coordinates of the winners are then recomputed from the vertices, with
gradients, so whatever is interpolated over the triangles (texture coordinates, depth) is
differentiable in the vertices and in the interpolated values. Coverage itself is not: a
triangle's edges get no gradient, which is fine where the mesh's place comes from elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

NEAR = 1e-3
"""Triangles with a corner nearer than this to the camera plane are skipped."""
MAX_CANDIDATES = 1 << 22
"""Pixels tested at once; bounds the memory a rasterization takes."""
EDGE_TOLERANCE = 1e-5
"""Pixels this close outside a triangle, in barycentric terms, count as inside, so rounding
leaves no cracks between neighbouring triangles."""
_NONE = torch.iinfo(torch.int64).max


@dataclass(frozen=True)
class Fragments:
    """The covered pixels of a rasterized image and the triangle point each one sees."""

    width: int
    height: int
    pixels: torch.Tensor
    """``(P,)`` flat indices ``y * width + x`` of the covered pixels."""
    faces: torch.Tensor
    """``(P,)`` the triangle each covered pixel sees."""
    barycentric: torch.Tensor
    """``(P, 3)`` perspective-correct barycentric coordinates of the seen point."""
    depth: torch.Tensor
    """``(P,)`` depth of the seen point."""

    def interpolate(self, corners: torch.Tensor) -> torch.Tensor:
        """``(P, C)`` values interpolated from ``(F, 3, C)`` values at the triangles' corners."""
        return (corners[self.faces] * self.barycentric.unsqueeze(-1)).sum(dim=1)

    def image(self, values: torch.Tensor, fill: float = 0.0) -> torch.Tensor:
        """``(H, W, C)`` image of ``(P, C)`` per-fragment values, ``fill`` where uncovered."""
        image = values.new_full((self.height * self.width, values.shape[-1]), fill)
        image = image.index_put((self.pixels,), values)
        return image.view(self.height, self.width, -1)

    def mask(self) -> torch.Tensor:
        """``(H, W)`` bool: which pixels are covered."""
        mask = torch.zeros(self.height * self.width, dtype=torch.bool, device=self.pixels.device)
        mask[self.pixels] = True
        return mask.view(self.height, self.width)

    def depth_image(self) -> torch.Tensor:
        """``(H, W)`` depth, infinite where uncovered."""
        return self.image(self.depth.unsqueeze(-1), float("inf"))[..., 0]


def rasterize(points: torch.Tensor, faces: torch.Tensor, width: int, height: int) -> Fragments:
    """Rasterize triangles given in screen space.

    Args:
        points: ``(V, 3)`` vertices as x, y in pixels (pixel centres at half-integers, as in
            gsplat) and depth along the view direction. For a flat layout such as a texture
            atlas, give every vertex depth 1.
        faces: ``(F, 3)`` triangles; both windings are drawn.
    """
    with torch.no_grad():
        nearest = _nearest(points, faces, width, height)
    pixels = torch.nonzero(nearest != _NONE).squeeze(-1)
    seen = (nearest[pixels] & 0xFFFFFFFF).long()

    centres = torch.stack((pixels % width, pixels // width), dim=-1).to(points) + 0.5
    corners = points[faces[seen]]
    screen = _barycentric(corners[..., :2], centres)
    weights = screen / corners[..., 2]
    inverse_depth = weights.sum(dim=-1)
    return Fragments(
        width=width,
        height=height,
        pixels=pixels,
        faces=seen,
        barycentric=weights / inverse_depth.unsqueeze(-1),
        depth=1 / inverse_depth,
    )


def _nearest(points: torch.Tensor, faces: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """``(H * W,)`` int64 per pixel: the nearest depth's float bits, shifted up 32 bits, or'ed
    with the index of the triangle at that depth; ``_NONE`` where no triangle covers it."""
    corners = points[faces]
    xy, z = corners[..., :2], corners[..., 2]
    first = (xy.min(dim=1).values - 0.5).ceil().clamp_min(0).long()
    last = (xy.max(dim=1).values - 0.5).floor()
    last = torch.minimum(last, last.new_tensor([width - 1, height - 1])).long()
    a, b, c = xy.unbind(dim=1)
    area = _cross(b - a, c - a)
    drawn = (z > NEAR).all(dim=1) & (last >= first).all(dim=1) & (area != 0)
    extent = (last - first + 1).amax(dim=1)

    nearest = torch.full((height * width,), _NONE, dtype=torch.int64, device=points.device)
    size = 1
    while drawn.any():
        # Triangles are tested in groups of similar size, each against a square of pixels.
        group = torch.nonzero(drawn & (extent <= size)).squeeze(-1)
        drawn[group] = False
        grid = torch.stack(
            torch.meshgrid(torch.arange(size), torch.arange(size), indexing="xy"), dim=-1
        )
        grid = grid.view(-1, 2).to(points.device)
        for chunk in group.split(max(1, MAX_CANDIDATES // size**2)):
            pixel = first[chunk, None] + grid
            inside = (pixel <= last[chunk, None]).all(dim=-1)
            weights = _barycentric(xy[chunk, None], pixel.to(points) + 0.5)
            inside &= (weights >= -EDGE_TOLERANCE).all(dim=-1)
            depth = 1 / (weights / z[chunk, None]).sum(dim=-1)
            key = (depth.float().view(torch.int32).long() << 32) | chunk[:, None]
            index = pixel[..., 1] * width + pixel[..., 0]
            nearest.scatter_reduce_(0, index[inside], key[inside], "amin")
        size *= 2
    return nearest


def _barycentric(triangles: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """``(..., 3)`` barycentric coordinates of ``(..., 2)`` points in ``(..., 3, 2)`` triangles,
    whatever their winding."""
    a, b, c = triangles.unbind(dim=-2)
    area = _cross(b - a, c - a)
    weights = torch.stack(
        (_cross(c - b, points - b), _cross(a - c, points - c), _cross(b - a, points - a)), dim=-1
    )
    return weights / area.unsqueeze(-1)


def _cross(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
