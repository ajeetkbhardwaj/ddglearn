"""Curl operators in Discrete Exterior Calculus.

In 2D DEC:

- Scalar Laplacian:
    Δu = *0^{-1} d0^T *1 d0 u

- Vector curl (circulation per face):
    curl v = d1 v

Note:
There is no scalar-to-scalar curl in 2D.
The scalar case computes the Laplace–Beltrami operator.
"""
import numpy as np
from typing import Any

from operators.exterior_derivative import d1

try:
    from scipy.sparse import isspmatrix, csr_matrix
    _HAS_SCIPY = True
except Exception:
    isspmatrix = lambda x: False
    _HAS_SCIPY = False


def curl_scalar(mesh, u: np.ndarray) -> Any:
    """Compute Laplace–Beltrami operator of a scalar field."""
    from operators.laplacian import laplacian_0

    L = laplacian_0(mesh)

    if _HAS_SCIPY and isspmatrix(L):
        L = csr_matrix(L)
        result = L @ u
        return np.asarray(result).reshape(-1)
    else:
        L = L.toarray() if hasattr(L, "toarray") else np.asarray(L)
        return L @ u


def curl_vector(mesh, v: np.ndarray) -> Any:
    """Compute curl (circulation) of a 1-form.

    Maps 1-form (edge values) to 2-form (face values).
    Returns: d1 @ v
    """
    D1 = d1(mesh)

    if _HAS_SCIPY and isspmatrix(D1):
        D1 = csr_matrix(D1)
        result = D1 @ v
        return np.asarray(result).reshape(-1)
    else:
        D1 = D1.toarray() if hasattr(D1, "toarray") else np.asarray(D1)
        return D1 @ v


__all__ = ["curl_scalar", "curl_vector"]