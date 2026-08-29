"""Heat Kernel Signature (HKS) utilities."""
import numpy as np
from typing import Tuple

from .eigen import eigen_decomposition


def compute_hks(mesh, k: int = 50, times: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """Compute HKS for `mesh` using first `k` Laplacian eigenpairs.

    Returns (times, hks) where `hks` has shape (n_vertices, len(times)).
    """
    vals, vecs = eigen_decomposition(mesh, k=k)
    n = mesh.n_vertices
    m = vecs.shape[1]

    # choose times if not provided: logspace between small and large scales
    if times is None:
        # avoid zero eigenvalue for scale choice
        positive = vals[vals > 1e-12]
        if positive.size == 0:
            tmin, tmax = 1e-3, 1.0
        else:
            tmin = 1.0 / (positive[-1] + 1e-12) * 1e-3
            tmax = 1.0 / (positive[0] + 1e-12) * 1e2 if positive[0] > 0 else 1.0
        times = np.logspace(np.log10(tmin), np.log10(tmax), num=10)

    # compute squared eigenfunctions
    phi2 = vecs ** 2  # shape (n, m)

    # Vectorized over time: exp_factors has shape (len(times), m).
    exp_factors = np.exp(-np.outer(times, vals))  # (T, m)
    hks = phi2 @ exp_factors.T  # (n, T)

    return times, hks


__all__ = ["compute_hks"]
