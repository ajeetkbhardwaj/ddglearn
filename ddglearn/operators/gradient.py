"""Gradient operator: vertex scalar → edge values.

The gradient of a scalar field u at vertices maps to 1-form values (edge scalars).
Using DEC: grad u = *0^{-1} d0^T *1 u  (but we return the more intuitive d0^T u)
"""
import numpy as np

from ..operators.exterior_derivative import d0
from ..operators.hodge_star import hodge_star_0, hodge_star_1

try:
    from scipy.sparse import isspmatrix, csr_matrix
    _HAS_SCIPY = True
except Exception:
    isspmatrix = lambda x: False
    _HAS_SCIPY = False


def gradient(mesh, u: np.ndarray):
    if u.shape[0] != mesh.n_vertices:
        raise ValueError("Input scalar field must have size n_vertices")

    D0 = d0(mesh)

    if _HAS_SCIPY and isspmatrix(D0):
        D0 = csr_matrix(D0)
        return np.asarray(D0 @ u).reshape(-1)
    else:
        D0_dense = D0.toarray() if hasattr(D0, "toarray") else np.asarray(D0)
        return D0_dense @ u


def gradient_vector(mesh, u: np.ndarray) -> np.ndarray:
    """Compute the 3D gradient vector on each face.

    Returns: (n_faces, 3) array of vectors.
    """
    if u.shape[0] != mesh.n_vertices:
        raise ValueError("Input scalar field must have size n_vertices")

    V = mesh.vertices
    F = mesh.faces

    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    u0, u1, u2 = u[F[:, 0]], u[F[:, 1]], u[F[:, 2]]

    # Normal to face
    face_normals = np.cross(v1 - v0, v2 - v0)
    area2 = np.linalg.norm(face_normals, axis=1, keepdims=True)
    area2 = np.maximum(area2, 1e-12)
    n = face_normals / area2

    # e0 opposite to v0, e1 to v1, e2 to v2
    e0, e1, e2 = v2 - v1, v0 - v2, v1 - v0

    grad = (
        u0[:, None] * np.cross(n, e0)
        + u1[:, None] * np.cross(n, e1)
        + u2[:, None] * np.cross(n, e2)
    ) / area2

    return grad


__all__ = ["gradient", "gradient_vector"]
