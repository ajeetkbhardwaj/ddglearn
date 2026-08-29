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


def gaussian_curvature(mesh) -> np.ndarray:
    V = mesh.vertices
    F = mesh.faces
    n_v = mesh.n_vertices

    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    e01, e02 = v1 - v0, v2 - v0
    e10, e12 = v0 - v1, v2 - v1
    e20, e21 = v0 - v2, v1 - v2

    angle0 = np.arctan2(np.linalg.norm(np.cross(e01, e02), axis=1), np.sum(e01 * e02, axis=1))
    angle1 = np.arctan2(np.linalg.norm(np.cross(e10, e12), axis=1), np.sum(e10 * e12, axis=1))
    angle2 = np.arctan2(np.linalg.norm(np.cross(e20, e21), axis=1), np.sum(e20 * e21, axis=1))

    angle_sum = np.zeros(n_v, dtype=float)
    np.add.at(angle_sum, F[:, 0], angle0)
    np.add.at(angle_sum, F[:, 1], angle1)
    np.add.at(angle_sum, F[:, 2], angle2)

    A = mesh.vertex_area_voronoi()
    boundary_indices = [v for e in mesh.boundary_edges() for v in e]

    A_safe = np.maximum(A, 1e-14)
    is_boundary = np.zeros(n_v, dtype=bool)
    if boundary_indices:
        is_boundary[np.array(list(set(boundary_indices)), dtype=int)] = True
    deficit = np.where(is_boundary, np.pi - angle_sum, 2.0 * np.pi - angle_sum)
    return deficit / A_safe


def mean_curvature_vector(mesh) -> np.ndarray:
    """Compute discrete mean curvature vector Hn at each vertex.

    Uses cotangent weights via the cotangent_laplacian operator. The
    convention here gives Hn = L @ V / A (no extra negation).
    """
    from operators.cotangent_laplacian import cotangent_laplacian

    V = mesh.vertices

    A = mesh.vertex_area_voronoi()

    L = cotangent_laplacian(mesh)

    A_safe = np.maximum(A, 1e-12)
    if hasattr(L, "dot"):
        return L.dot(V) / A_safe[:, None]
    else:
        return (L @ V) / A_safe[:, None]


def mean_curvature(mesh) -> np.ndarray:
    """Return scalar mean curvature per vertex (magnitude of mean curvature vector).

    The discrete mean curvature vector `Hn` approximates `2 H n` in the smooth
    setting; here we return the scalar mean curvature defined as 0.5 * ||Hn||
    so that it corresponds to the usual `H` when `Hn` approximates 2 H n.
    """
    Hn = mean_curvature_vector(mesh)
    return 0.5 * np.linalg.norm(Hn, axis=1)


def principal_curvatures(mesh):
    K = gaussian_curvature(mesh)
    H = mean_curvature(mesh)

    discr = np.maximum(H * H - K, 0.0)
    sqrt_discr = np.sqrt(discr)
    k1 = H + sqrt_discr
    k2 = H - sqrt_discr

    return k1, k2

__all__ = [
    "gaussian_curvature",
    "mean_curvature_vector",
    "mean_curvature",
    "principal_curvatures",
]
