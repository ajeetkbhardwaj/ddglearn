"""Cotangent Laplacian operator.

Provides the discrete Laplace-Beltrami operator using the cotangent formula directly.
"""

import numpy as np

try:
    from scipy.sparse import coo_matrix
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def cotangent_laplacian(mesh):
    """Construct the cotangent Laplace-Beltrami operator matrix L.

    Args:
        mesh: The HalfEdgeMesh instance.

    Returns:
        L: (n_vertices, n_vertices) SciPy sparse matrix (if installed) or
           dense numpy array. Off-diagonal entries are non-positive and the
        diagonal is the positive sum of adjacent cotangent weights, so L is
        positive semi-definite and consistent with laplacian_0.
    """
    # Check cache
    cache = getattr(mesh, "_cache", None)
    if isinstance(cache, dict) and "cotan_laplacian" in cache:
        return cache["cotan_laplacian"]
    if hasattr(mesh, "_cache") and hasattr(mesh._cache, "operators"):
        if "cotan_laplacian" in mesh._cache.operators:
            return mesh._cache.operators["cotan_laplacian"]

    V = mesh.vertices
    F = mesh.faces
    n_v = mesh.n_vertices

    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]

    e0, e1, e2 = v2 - v1, v0 - v2, v1 - v0

    cross0, cross1, cross2 = np.cross(e1, -e2), np.cross(e2, -e0), np.cross(e0, -e1)
    norm0 = np.maximum(np.linalg.norm(cross0, axis=1), 1e-12)
    norm1 = np.maximum(np.linalg.norm(cross1, axis=1), 1e-12)
    norm2 = np.maximum(np.linalg.norm(cross2, axis=1), 1e-12)

    cot0 = 0.5 * np.sum(e1 * (-e2), axis=1) / norm0
    cot1 = 0.5 * np.sum(e2 * (-e0), axis=1) / norm1
    cot2 = 0.5 * np.sum(e0 * (-e1), axis=1) / norm2

    I = np.concatenate([F[:, 1], F[:, 2], F[:, 1], F[:, 2], F[:, 2], F[:, 0], F[:, 2], F[:, 0], F[:, 0], F[:, 1], F[:, 0], F[:, 1]])
    J = np.concatenate([F[:, 2], F[:, 1], F[:, 1], F[:, 2], F[:, 0], F[:, 2], F[:, 2], F[:, 0], F[:, 1], F[:, 0], F[:, 0], F[:, 1]])
    # Standard sign convention: negative on off-diagonal, positive on diagonal.
    vals = np.concatenate([
        -0.5 * cot0, -0.5 * cot0, 0.5 * cot0, 0.5 * cot0,
        -0.5 * cot1, -0.5 * cot1, 0.5 * cot1, 0.5 * cot1,
        -0.5 * cot2, -0.5 * cot2, 0.5 * cot2, 0.5 * cot2,
    ])

    if _HAS_SCIPY:
        L = coo_matrix((vals, (I, J)), shape=(n_v, n_v)).tocsr()
    else:
        L = np.zeros((n_v, n_v), dtype=float)
        np.add.at(L, (I, J), vals)

    # Store in cache
    if isinstance(cache, dict):
        cache["cotan_laplacian"] = L
    elif hasattr(mesh, "_cache") and hasattr(mesh._cache, "operators"):
        mesh._cache.operators["cotan_laplacian"] = L

    return L

__all__ = ["cotangent_laplacian"]
