"""DEC Laplacian assembly for 0-forms (vertex scalar Laplacian).

Returns the **Weak Laplacian** (Stiffness Matrix):

    L_c = d0^T *1 d0

This is symmetric negative semi-definite.  To obtain the Strong
Laplacian (pointwise operator) divide by the mass matrix on the
**right-hand side** when solving:  L_c x = M b.

Never use the strong form M^{-1} L_c directly in a linear solve —
always keep the mass matrix on the RHS to preserve symmetry.
"""
import numpy as np
from typing import Any

from .exterior_derivative import d0
from .hodge_star import hodge_star_1

try:
    from scipy.sparse import isspmatrix, csr_matrix
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def laplacian_0(mesh) -> Any:
    """Assemble the Weak Laplacian (Stiffness Matrix) for 0-forms.

    L_c = d0^T *1 d0  —  symmetric negative semi-definite.

    Returns sparse (if scipy) or dense numpy array otherwise.
    """
    cache = getattr(mesh, "_cache", None)
    if isinstance(cache, dict) and "laplacian_0" in cache:
        return cache["laplacian_0"]
    if hasattr(mesh, "_cache") and hasattr(mesh._cache, "operators"):
        if "laplacian_0" in mesh._cache.operators:
            return mesh._cache.operators["laplacian_0"]

    D0 = d0(mesh)
    H1 = hodge_star_1(mesh)

    if _HAS_SCIPY:
        if not isspmatrix(D0):
            D0 = csr_matrix(D0)
        if not isspmatrix(H1):
            H1 = csr_matrix(H1)

        L = (D0.T @ H1 @ D0).tocsr()
    else:
        D0 = D0.toarray() if hasattr(D0, "toarray") else np.asarray(D0)
        H1 = H1.toarray() if hasattr(H1, "toarray") else np.asarray(H1)
        L = D0.T @ H1 @ D0

    # Store in cache
    if isinstance(cache, dict):
        cache["laplacian_0"] = L
    elif hasattr(mesh, "_cache") and hasattr(mesh._cache, "operators"):
        mesh._cache.operators["laplacian_0"] = L
        
    return L


__all__ = ["laplacian_0"]
