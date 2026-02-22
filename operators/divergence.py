"""Divergence operator: edge vector field → vertex scalars.

Divergence takes a 1-form (edge values) and maps to 0-form (vertex scalars).
Using DEC: div v = -d1^T *2^{-1} v  (or equivalently -d1^T v / face_areas)
"""
import numpy as np
from typing import Any

from operators.exterior_derivative import d0
from operators.hodge_star import hodge_star_0, hodge_star_1

try:
    from scipy.sparse import isspmatrix, csr_matrix, diags
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

        w = H1.dot(v)
        div = -D0.T.dot(w)

        diag = H0.diagonal()
        eps = np.finfo(float).eps
        inv_diag = 1.0 / np.maximum(diag, eps)

        from scipy.sparse import diags
        H0_inv = diags(inv_diag)

        result = H0_inv.dot(div)
        return np.asarray(result).reshape(-1)

    else:
        D0 = np.asarray(D0)
        H0 = np.asarray(H0)
        H1 = np.asarray(H1)

        w = H1.dot(v)
        div = -D0.T.dot(w)

        diag = np.diag(H0)
        eps = np.finfo(float).eps
        inv_diag = 1.0 / np.maximum(diag, eps)

        return np.diag(inv_diag).dot(div)

__all__ = ["divergence"]
