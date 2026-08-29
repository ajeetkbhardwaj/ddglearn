"""Divergence operator: edge vector field → vertex scalars.

Divergence takes a 1-form (edge values) and maps to 0-form (vertex scalars).
Using DEC: div v = -*0^{-1} d0^T *1 v
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




def divergence(mesh, v: np.ndarray):
    D0 = d0(mesh)
    H0 = hodge_star_0(mesh)
    H1 = hodge_star_1(mesh)

    if v.shape[0] != D0.shape[0]:
        raise ValueError("Input 1-form must have size n_edges")

    if _HAS_SCIPY:
        if not isspmatrix(D0):
            D0 = csr_matrix(D0)
        if not isspmatrix(H0):
            H0 = csr_matrix(H0)
        if not isspmatrix(H1):
            H1 = csr_matrix(H1)

        w = H1 @ v
        div = -D0.T @ w

        diag = H0.diagonal()
        eps = np.finfo(float).eps
        inv_diag = 1.0 / np.maximum(diag, eps)

        from scipy.sparse import diags
        H0_inv = diags(inv_diag)

        result = H0_inv @ div
        return np.asarray(result).reshape(-1)

    else:
        D0 = D0.toarray() if hasattr(D0, "toarray") else np.asarray(D0)
        H0 = H0.toarray() if hasattr(H0, "toarray") else np.asarray(H0)
        H1 = H1.toarray() if hasattr(H1, "toarray") else np.asarray(H1)

        w = H1 @ v
        div = -D0.T @ w

        diag = np.diag(H0)
        eps = np.finfo(float).eps
        inv_diag = 1.0 / np.maximum(diag, eps)

        return np.diag(inv_diag) @ div


def divergence_face_vector(mesh, X: np.ndarray) -> np.ndarray:
    """Compute divergence of a constant-per-face vector field X.

    Args:
        mesh: HalfEdgeMesh instance
        X: (n_faces, 3) array of vectors

    Returns: (n_vertices,) array of divergence values
    """
    V = mesh.vertices
    F = mesh.faces
    n_v = mesh.n_vertices

    div = np.zeros(n_v)

    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    e0, e1, e2 = v2 - v1, v0 - v2, v1 - v0

    # Face normals (magnitude is 2 * area)
    n = np.cross(v1 - v0, v2 - v0)

    # div_i = -0.5 * sum_j X_j . (n_j x e_ij)
    cot_part0 = -0.5 * np.sum(X * np.cross(n, e0), axis=1)
    cot_part1 = -0.5 * np.sum(X * np.cross(n, e1), axis=1)
    cot_part2 = -0.5 * np.sum(X * np.cross(n, e2), axis=1)

    np.add.at(div, F[:, 0], cot_part0)
    np.add.at(div, F[:, 1], cot_part1)
    np.add.at(div, F[:, 2], cot_part2)

    return div


__all__ = ["divergence", "divergence_face_vector"]
