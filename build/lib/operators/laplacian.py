"""DEC Laplacian assembly for 0-forms (vertex scalar Laplacian).

We use the DEC expression

    W = d0^T *1 d0
    L0 = (*0)^{-1} W

which produces a symmetric positive-semidefinite operator on vertices.
"""
import numpy as np
from typing import Any

from .exterior_derivative import d0
from .hodge_star import hodge_star_0, hodge_star_1

try:
    from scipy.sparse import diags as _diags, isspmatrix, csr_matrix
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def laplacian_0(mesh) -> Any:
    """Assemble DEC Laplacian for 0-forms. Returns sparse (if scipy)
    or dense numpy array otherwise.
    """
    D0 = d0(mesh)
    H1 = hodge_star_1(mesh)
    H0 = hodge_star_0(mesh)

    if _HAS_SCIPY:
        # ensure CSR for fast products
        if not isspmatrix(D0):
            D0 = csr_matrix(D0)
        if not isspmatrix(H1):
            H1 = csr_matrix(H1)
        if not isspmatrix(H0):
            H0 = csr_matrix(H0)

        W = D0.T @ H1 @ D0
        # invert diagonal H0
        diag = H0.diagonal()
        eps = np.finfo(float).eps
        inv_diag = 1.0 / np.maximum(diag, eps)
        H0_inv = _diags(inv_diag)
        L = H0_inv @ W
        return L.tocsr()
    else:
        D0 = D0.toarray() if hasattr(D0, "toarray") else np.asarray(D0)
        H1 = H1.toarray() if hasattr(H1, "toarray") else np.asarray(H1)
        H0 = H0.toarray() if hasattr(H0, "toarray") else np.asarray(H0)

        W = D0.T @ H1 @ D0
        diag = np.diag(H0)
        eps = np.finfo(float).eps
        inv_diag = 1.0 / np.maximum(diag, eps)
        return np.diag(inv_diag) @ W


__all__ = ["laplacian_0"]
