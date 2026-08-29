"""Heat equation solvers (implicit Euler) using DEC assembly."""
import numpy as np
from typing import Optional

from ..operators.exterior_derivative import d0
from ..operators.hodge_star import hodge_star_1, hodge_star_0

try:
    from scipy.sparse import isspmatrix, csr_matrix
    from scipy.sparse.linalg import spsolve
    _HAS_SCIPY = True
except Exception:
    isspmatrix = lambda x: False
    spsolve = None
    _HAS_SCIPY = False


def implicit_heat_step(
    mesh,
    u: np.ndarray,
    t: float,
    pin_index: Optional[int] = 0,
    dirichlet_indices: Optional[np.ndarray] = None,
    dirichlet_values: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Perform a single implicit Euler heat step with time-step `t`.

    Solves (H0 - t W) u_next = H0 u
    """
    D0 = d0(mesh)
    H1 = hodge_star_1(mesh)
    H0 = hodge_star_0(mesh)

    if _HAS_SCIPY and isspmatrix(D0):
        D0 = csr_matrix(D0)
        H1 = csr_matrix(H1) if not isspmatrix(H1) else H1
        H0 = csr_matrix(H0) if not isspmatrix(H0) else H0
        W = D0.T @ H1 @ D0
        A = (H0 + t * W).tocsr()
        rhs = H0 @ u
        
        if dirichlet_indices is not None and dirichlet_values is not None:
            A = A.tolil()
            for idx, val in zip(dirichlet_indices, dirichlet_values):
                rhs -= A[:, idx].toarray().flatten() * val
                A[idx, :] = 0
                A[:, idx] = 0
                A[idx, idx] = 1.0
                rhs[idx] = val
            A = A.tocsr()
        elif pin_index is not None:
            A = A.tolil()
            A[pin_index, :] = 0
            A[:, pin_index] = 0
            A[pin_index, pin_index] = 1.0
            A = A.tocsr()
            rhs[pin_index] = 0.0
        u_next = spsolve(A, rhs)
        return np.asarray(u_next).reshape(-1)
    else:
        D0 = D0.toarray() if hasattr(D0, "toarray") else np.asarray(D0)
        H1 = H1.toarray() if hasattr(H1, "toarray") else np.asarray(H1)
        H0 = H0.toarray() if hasattr(H0, "toarray") else np.asarray(H0)
        W = D0.T @ H1 @ D0
        A = H0 + t * W
        rhs = H0 @ u
        
        if dirichlet_indices is not None and dirichlet_values is not None:
            A = A.copy()
            for idx, val in zip(dirichlet_indices, dirichlet_values):
                rhs -= A[:, idx] * val
                A[idx, :] = 0
                A[:, idx] = 0
                A[idx, idx] = 1.0
                rhs[idx] = val
        elif pin_index is not None:
            A = A.copy()
            A[pin_index, :] = 0
            A[:, pin_index] = 0
            A[pin_index, pin_index] = 1.0
            rhs[pin_index] = 0.0
        u_next = np.linalg.solve(A, rhs)
        return u_next.reshape(-1)


def implicit_heat(
    mesh,
    u0: np.ndarray,
    t: float,
    steps: int = 1,
    pin_index: Optional[int] = 0,
    dirichlet_indices: Optional[np.ndarray] = None,
    dirichlet_values: Optional[np.ndarray] = None,
) -> np.ndarray:
    u = u0.copy()
    for _ in range(steps):
        u = implicit_heat_step(mesh, u, t, pin_index=pin_index, dirichlet_indices=dirichlet_indices, dirichlet_values=dirichlet_values)
    return u


__all__ = ["implicit_heat", "implicit_heat_step"]
