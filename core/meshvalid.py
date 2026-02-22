# core/mesh_validation.py

import numpy as np


def validate_triangle_mesh(vertices, faces):
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=int)

    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("Only triangle meshes supported")

    n_vertices = vertices.shape[0]

    # Index bounds
    if np.any(faces < 0) or np.any(faces >= n_vertices):
        raise ValueError("Face indices out of bounds")

    # NaN vertices
    if np.isnan(vertices).any():
        raise ValueError("Mesh contains NaN vertices")

    # Degenerate triangles
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)

    eps = np.finfo(float).eps * 10
    if np.any(areas < eps):
        raise ValueError("Degenerate triangles detected")