"""Reading meshes and the geometry of their triangles."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def read_obj(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the vertices and faces of an OBJ file, fan-triangulating polygons.

    Returns:
        ``(V, 3)`` float32 vertices and ``(F, 3)`` int64 zero-based triangle indices.
    """
    vertices, triangles = [], []
    with open(path) as file:
        for line in file:
            if line.startswith("v "):
                vertices.append(line.split()[1:4])
            elif line.startswith("f "):
                corners = [int(corner.split("/")[0]) - 1 for corner in line.split()[1:]]
                for i in range(1, len(corners) - 1):
                    triangles.append((corners[0], corners[i], corners[i + 1]))
    return np.array(vertices, dtype=np.float32), np.array(triangles, dtype=np.int64)


def read_obj_uvs(path: str | Path) -> np.ndarray:
    """``(F, 3, 2)`` texture coordinates at the corners of an OBJ file's triangles, in the order
    ``read_obj`` gives the triangles."""
    coordinates, corners = [], []
    with open(path) as file:
        for line in file:
            if line.startswith("vt "):
                coordinates.append(line.split()[1:3])
            elif line.startswith("f "):
                polygon = [int(corner.split("/")[1]) - 1 for corner in line.split()[1:]]
                for i in range(1, len(polygon) - 1):
                    corners.append((polygon[0], polygon[i], polygon[i + 1]))
    return np.array(coordinates, dtype=np.float32)[np.array(corners)]


def read_obj_vertices(path: str | Path) -> np.ndarray:
    """Read only the ``(V, 3)`` vertices of an OBJ file."""
    with open(path) as file:
        rows = [line.split()[1:4] for line in file if line.startswith("v ")]
    return np.array(rows, dtype=np.float32)


def triangle_frames(
    vertices: torch.Tensor, faces: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Orthonormal frames and sizes of triangles, to attach things to a deforming surface.

    The frame's first axis runs along the triangle's first edge, its third axis is the normal.

    Args:
        vertices: ``(..., V, 3)`` vertex positions.
        faces: ``(F, 3)`` triangle indices.

    Returns:
        ``(..., F, 3, 3)`` rotations whose columns are the frame axes, and ``(..., F)`` sizes:
        the square root of twice the triangle area, a length that scales with the mesh.
    """
    a, b, c = (
        vertices[..., faces[:, 0], :],
        vertices[..., faces[:, 1], :],
        vertices[..., faces[:, 2], :],
    )
    edge = b - a
    normal = torch.linalg.cross(edge, c - a)
    size = normal.norm(dim=-1).clamp_min(1e-12).sqrt()
    x = torch.nn.functional.normalize(edge, dim=-1)
    z = torch.nn.functional.normalize(normal, dim=-1)
    y = torch.linalg.cross(z, x)
    return torch.stack((x, y, z), dim=-1), size


def uniform_laplacian(faces: torch.Tensor, num_vertices: int) -> torch.Tensor:
    """``(V, V)`` sparse matrix taking each vertex to its offset from the mean of its neighbours."""
    edges = torch.cat((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges = torch.unique(torch.cat((edges, edges.flip(1))), dim=0)
    degree = torch.bincount(edges[:, 0], minlength=num_vertices).clamp_min(1).float()
    indices = torch.cat((edges.T, torch.arange(num_vertices, device=faces.device).expand(2, -1)), 1)
    values = torch.cat((-1 / degree[edges[:, 0]], torch.ones(num_vertices, device=faces.device)))
    size = (num_vertices, num_vertices)
    return torch.sparse_coo_tensor(indices, values, size, check_invariants=True).coalesce()


def vertex_normals(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Area-weighted ``(..., V, 3)`` unit vertex normals."""
    a, b, c = (
        vertices[..., faces[:, 0], :],
        vertices[..., faces[:, 1], :],
        vertices[..., faces[:, 2], :],
    )
    face_normals = torch.linalg.cross(b - a, c - a)
    normals = torch.zeros_like(vertices)
    for corner in range(3):
        index = faces[:, corner].expand(*face_normals.shape[:-1])
        normals.scatter_add_(-2, index.unsqueeze(-1).expand_as(face_normals), face_normals)
    return torch.nn.functional.normalize(normals, dim=-1)


def boundary_loops(faces: np.ndarray) -> list[np.ndarray]:
    """The closed loops of edges that belong to one triangle only: a mesh's holes and cut edges.

    Each loop is a vertex sequence running the way its triangles wind, so for outward-facing
    triangles seen from outside, the hole is on the left. Where a vertex touches the boundary
    twice, the loops through it may come out joined.
    """
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    pairs = {tuple(edge) for edge in edges.tolist()}
    following = {a: b for a, b in pairs if (b, a) not in pairs}
    loops = []
    while following:
        start, vertex = next(iter(following.items()))
        loop = [start]
        del following[start]
        while vertex != start and vertex in following:
            loop.append(vertex)
            vertex = following.pop(vertex)
        loops.append(np.array(loop))
    return loops


def zip_loops(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Triangles bridging two loops of points that go once around the z axis, one above the other.

    Args:
        lower, upper: ``(N, 3)`` and ``(M, 3)`` points in order around the loops, either way.

    Returns:
        ``(N + M, 3)`` triangles indexing ``lower`` then ``upper`` (as ``N + j``), facing away from
        the axis through the loops' centre.
    """
    centre = lower.mean(axis=0)

    def angles(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The loop's order made to run anticlockwise seen from above, starting at angle 0, and
        the unwrapped angle of each point in that order."""
        angle = np.arctan2(points[:, 1] - centre[1], points[:, 0] - centre[0])
        order = np.arange(len(points))
        if np.sum(np.angle(np.exp(1j * np.diff(angle, append=angle[:1])))) < 0:
            order = order[::-1]
        order = np.roll(order, -int(np.argmin(np.mod(angle[order], 2 * np.pi))))
        unwrapped = np.unwrap(angle[order])
        return order, unwrapped - unwrapped[0] + np.mod(angle[order[0]], 2 * np.pi)

    low_order, low_angle = angles(lower)
    up_order, up_angle = angles(upper)
    n, m = len(lower), len(upper)
    low_angle = np.append(low_angle, low_angle[0] + 2 * np.pi)
    up_angle = np.append(up_angle, up_angle[0] + 2 * np.pi)

    triangles = []
    i = j = 0
    while i < n or j < m:
        a, a_next = low_order[i % n], low_order[(i + 1) % n]
        b, b_next = up_order[j % m], up_order[(j + 1) % m]
        if j == m or (i < n and low_angle[i + 1] <= up_angle[j + 1]):
            triangles.append((a, a_next, n + b))
            i += 1
        else:
            triangles.append((a, n + b_next, n + b))
            j += 1
    return np.array(triangles)


def closest_points(
    points: torch.Tensor, vertices: torch.Tensor, faces: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The nearest point of a mesh to each of ``(P, 3)`` points, by brute force: for small meshes.

    Returns:
        ``(P,)`` the triangle the nearest point is on, and ``(P, 3)`` its barycentric coordinates.
    """
    a, b, c = vertices[faces].unbind(dim=1)  # (F, 3) each
    p = points[:, None]  # (P, 1, 3)
    candidates = [plane_barycentric(p, a, b, c)]
    for start, end, corners in ((a, b, (0, 1)), (b, c, (1, 2)), (c, a, (2, 0))):
        edge = end - start
        t = (((p - start) * edge).sum(-1) / edge.square().sum(-1).clamp_min(1e-20)).clamp(0, 1)
        weights = torch.zeros(*t.shape, 3, device=points.device)
        weights[..., corners[0]] = 1 - t
        weights[..., corners[1]] = t
        candidates.append(weights)
    weights = torch.stack(candidates)  # (4, P, F, 3)
    nearest = (weights.unsqueeze(-1) * torch.stack((a, b, c), dim=1)).sum(dim=-2)
    distance = (nearest - p).square().sum(-1)  # (4, P, F)
    distance = torch.where(weights.min(dim=-1).values >= 0, distance, float("inf"))
    best = distance.permute(1, 0, 2).reshape(len(points), -1).argmin(dim=1)
    candidate, face = best // len(faces), best % len(faces)
    rows = torch.arange(len(points), device=points.device)
    return face, weights[candidate, rows, face]


def plane_barycentric(
    p: torch.Tensor, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    """``(..., 3)`` barycentric coordinates of ``(..., 3)`` points projected onto the planes of
    triangles with corners ``a``, ``b``, ``c``; some negative where the projection falls outside
    the triangle."""
    u, v, w = b - a, c - a, p - a
    uu, uv, vv = (u * u).sum(-1), (u * v).sum(-1), (v * v).sum(-1)
    wu, wv = (w * u).sum(-1), (w * v).sum(-1)
    denominator = (uu * vv - uv * uv).clamp_min(1e-20)
    beta = (vv * wu - uv * wv) / denominator
    gamma = (uu * wv - uv * wu) / denominator
    return torch.stack((1 - beta - gamma, beta, gamma), dim=-1)
