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
