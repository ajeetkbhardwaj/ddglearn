"""Optimal Transport Maps on Surfaces.

Implements Convolutional Wasserstein Distances (Solomon et al. 2015),
which uses the Heat Equation to approximate the Sinkhorn kernel convolution.
This computes optimal transport in O(N) time without an O(N^2) distance matrix.
"""
import numpy as np


def sinkhorn_wasserstein(mesh, p, q, t: float = 1e-3, max_iter: int = 150, tol: float = 1e-5):
    """Compute the Optimal Transport (Wasserstein) distance between distributions.

    Args:
        mesh (HalfEdgeMesh): The mesh.
        p (np.ndarray): Source mass distribution array (shape: n_vertices).
        q (np.ndarray): Target mass distribution array (shape: n_vertices).
        t (float): Diffusion time (controls entropy regularization / blurring).
        max_iter (int): Maximum Sinkhorn iterations.
        tol (float): Convergence tolerance.

    Returns:
        dist (float): Approximate Wasserstein distance.
        u (np.ndarray): Left dual scaling variable.
        v (np.ndarray): Right dual scaling variable.
    """
    from ..operators.exterior_derivative import d0
    from ..operators.hodge_star import hodge_star_0, hodge_star_1

    try:
        from scipy.sparse import csr_matrix, isspmatrix
        from scipy.sparse.linalg import factorized
        _LOCAL_HAS_SCIPY = True
    except Exception:
        _LOCAL_HAS_SCIPY = False

    D0 = d0(mesh)
    H0 = hodge_star_0(mesh)
    H1 = hodge_star_1(mesh)

    u = np.ones_like(p)
    v = np.ones_like(q)

    if _LOCAL_HAS_SCIPY:
        D0 = csr_matrix(D0) if not isspmatrix(D0) else D0
        H1 = csr_matrix(H1) if not isspmatrix(H1) else H1
        H0 = csr_matrix(H0) if not isspmatrix(H0) else H0

        W = D0.T @ H1 @ D0
        # Smoothing heat kernel: (H0 + t * W).
        A = (H0 + t * W).tocsc()
        solve_A = factorized(A)

        for _ in range(max_iter):
            Hv = solve_A(H0 @ v)
            u_new = p / np.maximum(Hv, 1e-12)

            Hu = solve_A(H0 @ u_new)
            v_new = q / np.maximum(Hu, 1e-12)

            err = np.max(np.abs(u_new - u))
            u, v = u_new, v_new

            if err < tol:
                break
    else:
        D0_dense = D0.toarray() if hasattr(D0, "toarray") else np.asarray(D0)
        H1_dense = H1.toarray() if hasattr(H1, "toarray") else np.asarray(H1)
        H0_dense = H0.toarray() if hasattr(H0, "toarray") else np.asarray(H0)
        W = D0_dense.T @ H1_dense @ D0_dense
        # Smoothing heat kernel: (H0 + t * W).
        A = H0_dense + t * W

        for _ in range(max_iter):
            Hv = np.linalg.solve(A, H0_dense @ v)
            u_new = p / np.maximum(Hv, 1e-12)

            Hu = np.linalg.solve(A, H0_dense @ u_new)
            v_new = q / np.maximum(Hu, 1e-12)

            err = np.max(np.abs(u_new - u))
            u, v = u_new, v_new

            if err < tol:
                break

    dist = t * np.sum(p * np.log(np.maximum(u, 1e-12)) + q * np.log(np.maximum(v, 1e-12)))
    return dist, u, v


__all__ = ["sinkhorn_wasserstein"]
