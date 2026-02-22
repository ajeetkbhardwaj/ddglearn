"""Curvature computations: Gaussian, mean (vector and scalar), principal curvatures.

Functions
- `gaussian_curvature(mesh)` -> (n_vertices,) array of Gaussian curvature (angle deficit / area)
- `mean_curvature_vector(mesh)` -> (n_vertices,3) mean curvature vector (discrete)
- `mean_curvature(mesh)` -> (n_vertices,) scalar mean curvature (magnitude)
- `principal_curvatures(mesh)` -> (k1,k2) arrays of principal curvatures per vertex

Notes
- Uses angle-deficit for Gaussian curvature and cotangent weights for mean curvature vector.
"""
import numpy as np
from typing import Tuple


def _face_areas_and_normals(mesh):
    faces = mesh.faces
    V = mesh.vertices
    n_f = mesh.n_faces
    areas = np.zeros(n_f, dtype=float)
    normals = np.zeros((n_f, 3), dtype=float)
    for fi, f in enumerate(faces):
        v0, v1, v2 = [V[int(i)] for i in f]
        n = np.cross(v1 - v0, v2 - v0)
        a = 0.5 * np.linalg.norm(n)
        areas[fi] = a
        if a > 0:
            normals[fi] = n / (2.0 * a)
        else:
            normals[fi] = np.array([0.0, 0.0, 0.0])
    return areas, normals


def gaussian_curvature(mesh) -> np.ndarray:
    V = mesh.vertices
    F = mesh.faces
    n_v = mesh.n_vertices

    angle_sum = np.zeros(n_v, dtype=float)

    for f in F:
        i, j, k = [int(x) for x in f]
        vi, vj, vk = V[i], V[j], V[k]

        def angle(a, b, c):
            u = b - a
            v = c - a
            return np.arctan2(
                np.linalg.norm(np.cross(u, v)),
                np.dot(u, v)
            )

        angle_sum[i] += angle(vi, vj, vk)
        angle_sum[j] += angle(vj, vk, vi)
        angle_sum[k] += angle(vk, vi, vj)

    A = mesh.vertex_area_voronoi()
    A_safe = np.maximum(A, 1e-14)

    boundary_vertices = set()
    for a, b in mesh.boundary_edges():
        boundary_vertices.add(a)
        boundary_vertices.add(b)

    K = np.zeros(n_v)
    for i in range(n_v):
        if i in boundary_vertices:
            deficit = np.pi - angle_sum[i]
        else:
            deficit = 2.0 * np.pi - angle_sum[i]
        K[i] = deficit / A_safe[i]

    return K

def mean_curvature_vector(mesh) -> np.ndarray:
    """Compute discrete mean curvature vector Hn at each vertex.

    Uses cotangent weights: Hn_i = (1 / (2 A_i)) * sum_j (cot alpha_ij + cot beta_ij) * (v_j - v_i)
    """
    V = mesh.vertices
    faces = mesh.faces
    n_v = mesh.n_vertices

    # accumulate cotangent weights per undirected edge
    edge_cot = {}
    for f in faces:
        i, j, k = [int(x) for x in f]
        vi, vj, vk = V[i], V[j], V[k]

        def cot(a, b, c):
            ba = b - a
            ca = c - a
            cross = np.cross(ba, ca)
            denom = np.linalg.norm(cross)
            if denom < 1e-12:
                return 0.0
            return np.dot(ba, ca) / denom

        cot_i = cot(vi, vj, vk)
        cot_j = cot(vj, vk, vi)
        cot_k = cot(vk, vi, vj)

        # edge opposite i is (j,k)
        key = (min(j, k), max(j, k))
        edge_cot[key] = edge_cot.get(key, 0.0) + cot_i
        key = (min(i, k), max(i, k))
        edge_cot[key] = edge_cot.get(key, 0.0) + cot_j
        key = (min(i, j), max(i, j))
        edge_cot[key] = edge_cot.get(key, 0.0) + cot_k

    # build adjacency list of weights
    neigh_weights = {i: {} for i in range(n_v)}
    for (a, b), w in edge_cot.items():
        neigh_weights[a][b] = w
        neigh_weights[b][a] = w

    A = mesh.vertex_area_voronoi()
    A_safe = A.copy()
    A_safe[A_safe == 0] = 1e-12

    Hn = np.zeros((n_v, 3), dtype=float)
    for i in range(n_v):
        vi = V[i]
        s = np.zeros(3, dtype=float)
        for j, w in neigh_weights[i].items():
            vj = V[j]
            s += w * (vj - vi)
        Hn[i] = s / (2.0 * A_safe[i])
    return Hn


def mean_curvature(mesh) -> np.ndarray:
    """Return scalar mean curvature per vertex (magnitude of mean curvature vector).

    The discrete mean curvature vector `Hn` approximates `2 H n` in the smooth
    setting; here we return the scalar mean curvature defined as 0.5 * ||Hn||
    so that it corresponds to the usual `H` when `Hn` approximates 2 H n.
    """
    Hn = mean_curvature_vector(mesh)
    H_scalar = 0.5 * np.linalg.norm(Hn, axis=1)
    return H_scalar


def principal_curvatures(mesh):
    K = gaussian_curvature(mesh)
    H = mean_curvature(mesh)

    discr = np.maximum(H * H - K, 0.0)
    sqrt = np.sqrt(discr)

    k1 = H + sqrt
    k2 = H - sqrt

    return k1, k2

__all__ = [
    "gaussian_curvature",
    "mean_curvature_vector",
    "mean_curvature",
    "principal_curvatures",
]
