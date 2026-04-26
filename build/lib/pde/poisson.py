"""Poisson solver using DEC assembly.

Solves W u = H0 f where W = d0^T *1 d0 and H0 is the 0-form Hodge.
"""
import numpy as np
from typing import Optional

from operators.exterior_derivative import d0
from operators.hodge_star import hodge_star_1, hodge_star_0

try:
    from scipy.sparse import isspmatrix, csr_matrix
    from scipy.sparse.linalg import spsolve
    _HAS_SCIPY = True
except Exception:
    isspmatrix = lambda x: False
    spsolve = None
    _HAS_SCIPY = False


def solve_poisson(
    mesh,
    f: np.ndarray,
    pin_index: Optional[int] = 0,
    pin_value: float = 0.0,
    dirichlet_indices: Optional[np.ndarray] = None,
    dirichlet_values: Optional[np.ndarray] = None,
    neumann_indices: Optional[np.ndarray] = None,
    neumann_values: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Solve Poisson equation on mesh for given right-hand side `f` (vertex-wise).

    Returns solution `u` at vertices. By default pins `pin_index` to `pin_value`
    to make the linear system non-singular. Optionally accepts Dirichlet and
    Neumann boundary conditions.
    """
    D0 = d0(mesh)
    H1 = hodge_star_1(mesh)
    H0 = hodge_star_0(mesh)

    if _HAS_SCIPY and isspmatrix(D0):
        D0 = csr_matrix(D0)
        H1 = csr_matrix(H1) if not isspmatrix(H1) else H1
        H0 = csr_matrix(H0) if not isspmatrix(H0) else H0
        W = D0.T @ H1 @ D0
        rhs = H0 @ f
        
        if neumann_indices is not None and neumann_values is not None:
            rhs[neumann_indices] += neumann_values
            
        if dirichlet_indices is not None and dirichlet_values is not None:
            W = W.tolil()
            for idx, val in zip(dirichlet_indices, dirichlet_values):
                rhs -= W[:, idx].toarray().flatten() * val
                W[idx, :] = 0
                W[:, idx] = 0
                W[idx, idx] = 1.0
                rhs[idx] = val
            W = W.tocsr()
        elif pin_index is not None:
            W = W.tolil()
            rhs -= W[:, pin_index].toarray().flatten() * pin_value
            W[pin_index, :] = 0
            W[:, pin_index] = 0
            W[pin_index, pin_index] = 1.0
            W = W.tocsr()
            rhs[pin_index] = pin_value
        u = spsolve(W, rhs)
        return np.asarray(u).reshape(-1)
    else:
        D0 = D0.toarray() if hasattr(D0, "toarray") else np.asarray(D0)
        H1 = H1.toarray() if hasattr(H1, "toarray") else np.asarray(H1)
        H0 = H0.toarray() if hasattr(H0, "toarray") else np.asarray(H0)
        W = D0.T @ H1 @ D0
        rhs = H0 @ f
        
        if neumann_indices is not None and neumann_values is not None:
            rhs[neumann_indices] += neumann_values
            
        if dirichlet_indices is not None and dirichlet_values is not None:
            W = W.copy()
            for idx, val in zip(dirichlet_indices, dirichlet_values):
                rhs -= W[:, idx] * val
                W[idx, :] = 0
                W[:, idx] = 0
                W[idx, idx] = 1.0
                rhs[idx] = val
        elif pin_index is not None:
            W = W.copy()
            rhs -= W[:, pin_index] * pin_value
            W[pin_index, :] = 0
            W[:, pin_index] = 0
            W[pin_index, pin_index] = 1.0
            rhs[pin_index] = pin_value
        u = np.linalg.solve(W, rhs)
        return u.reshape(-1)


__all__ = ["solve_poisson"]
