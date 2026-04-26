"""Wave equation time-stepping using central differences.

Discretization: A (u^{n+1} - 2 u^n + u^{n-1}) / dt^2 + W u^n = 0
so A u^{n+1} = 2 A u^n - A u^{n-1} - dt^2 W u^n
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


def wave_step(mesh, u_prev: np.ndarray, u_curr: np.ndarray, dt: float, pin_index: Optional[int] = 0) -> np.ndarray:
    D0 = d0(mesh)
    H1 = hodge_star_1(mesh)
    H0 = hodge_star_0(mesh)

    if _HAS_SCIPY and isspmatrix(D0):
        D0 = csr_matrix(D0)
        H1 = csr_matrix(H1) if not isspmatrix(H1) else H1
        H0 = csr_matrix(H0) if not isspmatrix(H0) else H0
        W = D0.T @ H1 @ D0
        rhs = 2.0 * (H0 @ u_curr) - (H0 @ u_prev) - (dt * dt) * (W @ u_curr)
        A = H0.tocsr()
        if pin_index is not None:
            A = A.tolil()
            A[pin_index, :] = 0
            A[:, pin_index] = 0
            A[pin_index, pin_index] = 1.0
            A = A.tocsr()
            rhs = np.asarray(rhs).reshape(-1)
            rhs[pin_index] = 0.0
        u_next = spsolve(A, rhs)
        return np.asarray(u_next).reshape(-1)
    else:
        D0 = D0.toarray() if hasattr(D0, "toarray") else np.asarray(D0)
        H1 = H1.toarray() if hasattr(H1, "toarray") else np.asarray(H1)
        H0 = H0.toarray() if hasattr(H0, "toarray") else np.asarray(H0)
        W = D0.T @ H1 @ D0
        rhs = 2.0 * (H0 @ u_curr) - (H0 @ u_prev) - (dt * dt) * (W @ u_curr)
        A = np.asarray(H0)
        if pin_index is not None:
            A = A.copy()
            A[pin_index, :] = 0
            A[:, pin_index] = 0
            A[pin_index, pin_index] = 1.0
            rhs = np.asarray(rhs).reshape(-1)
            rhs[pin_index] = 0.0
        u_next = np.linalg.solve(A, rhs)
        return u_next.reshape(-1)


def simulate_wave(mesh, u0: np.ndarray, dt: float, steps: int, u1: Optional[np.ndarray] = None, pin_index: Optional[int] = 0) -> np.ndarray:
    # initialize u_prev and u_curr
    if u1 is None:
        u_prev = u0.copy()
        u_curr = u0.copy()
    else:
        u_prev = u0.copy()
        u_curr = u1.copy()

    for _ in range(steps):
        u_next = wave_step(mesh, u_prev, u_curr, dt, pin_index=pin_index)
        u_prev, u_curr = u_curr, u_next
    return u_curr


__all__ = ["wave_step", "simulate_wave"]
