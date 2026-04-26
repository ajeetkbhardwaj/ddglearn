"""Divergence operator: edge vector field → vertex scalars.

Divergence takes a 1-form (edge values) and maps to 0-form (vertex scalars).
Using DEC: div v = -*0^{-1} d0^T *1 v
"""
import numpy as np

from operators.exterior_derivative import d0
from operators.hodge_star import hodge_star_0, hodge_star_1

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

__all__ = ["divergence"]
