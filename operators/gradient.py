"""Gradient operator: vertex scalar → edge values.

The gradient of a scalar field u at vertices maps to 1-form values (edge scalars).
Using DEC: grad u = *0^{-1} d0^T *1 u  (but we return the more intuitive d0^T u)
"""
import numpy as np
from typing import Any

from operators.exterior_derivative import d0
from operators.hodge_star import hodge_star_0, hodge_star_1

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
        return np.asarray(D0.dot(u)).reshape(-1)
    else:
        return np.asarray(D0).dot(u)

__all__ = ["gradient"]
