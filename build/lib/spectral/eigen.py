"""Eigen decomposition utilities for mesh Laplacian."""

import numpy as np
from typing import Tuple

try:
    from scipy.sparse.linalg import eigsh
    from scipy.sparse import issparse

    _HAS_SCIPY = True
except Exception:
    eigsh = None
    issparse = lambda x: False
    _HAS_SCIPY = False


def eigen_decomposition(mesh, k: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the first `k` eigenpairs of the 0-form Laplacian.

    Returns (eigenvalues, eigenvectors) where eigenvectors are shape (n_vertices, k).
    """
    from operators.laplacian import laplacian_0

    L = laplacian_0(mesh)
    n = mesh.n_vertices
    k = min(k, max(1, n - 1))

    if _HAS_SCIPY and issparse(L):
        try:
            k_eff = min(k, n - 1)
            if k_eff <= 0:
                k_eff = 1
            vals, vecs = eigsh(L, k=k_eff, which="SM", maxiter=1000)
            idx = np.argsort(vals)
            vals = vals[idx]
            vecs = vecs[:, idx]
            return vals, vecs
        except Exception:
            pass

    Ld = np.asarray(L.toarray()) if hasattr(L, "toarray") else np.asarray(L)
    vals_all, vecs_all = np.linalg.eigh(Ld)
    idx = np.argsort(vals_all)
    vals = vals_all[idx][:k]
    vecs = vecs_all[:, idx][:, :k]
    return vals, vecs


__all__ = ["eigen_decomposition"]
