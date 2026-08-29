r"""Wave Kernel Signature (WKS): wavelet-based shape descriptor.

WKS is similar to HKS but uses a different exponential weighting function
and is often better at capturing fine-grained shape details.

Formula: WKS_t(x) = \sum_i e^{i\sqrt{\lambda_i}t} |\phi_i(x)|^2

This is a complex-valued alternative to HKS, effectively a Fourier transform
of the spectral density function.
"""

import numpy as np
from typing import Tuple

from ..spectral.eigen import eigen_decomposition


def compute_wks(mesh, k: int = 100, times: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """Compute WKS for mesh using first k Laplacian eigenpairs.

    Follows the definition from Aubry et al. 2011:
    WKS(x, e) = C(e) * sum_k exp(-(e - log(lambda_k))^2 / (2 * sigma^2)) * phi_k(x)^2

    Returns (log_energies, wks) where wks has shape (n_vertices, len(log_energies)).
    """
    vals, vecs = eigen_decomposition(mesh, k=k)
    
    # Ignore zero eigenvalue for log scale
    mask = vals > 1e-12
    vals = vals[mask]
    vecs = vecs[:, mask]
    
    log_vals = np.log(vals)
    phi2 = vecs**2

    if times is None:
        tmin = np.min(log_vals)
        tmax = np.max(log_vals)
        # standard heuristic for sigma
        sigma = (tmax - tmin) / 100.0 * 7.0
        times = np.linspace(tmin, tmax, num=100)
    else:
        # if times are provided, assume they are log-energies
        tmin = np.min(log_vals)
        tmax = np.max(log_vals)
        sigma = (tmax - tmin) / len(times) * 7.0

    wks = np.zeros((mesh.n_vertices, len(times)), dtype=float)

    # Vectorized over energy levels: weights has shape (len(times), m).
    diff = times[:, None] - log_vals[None, :]  # (T, m)
    weights = np.exp(-(diff ** 2) / (2 * sigma ** 2))  # (T, m)
    denom = weights.sum(axis=1)  # (T,)
    wks = (phi2 @ weights.T) / denom[None, :]  # (n, T)

    return times, wks


__all__ = ["compute_wks"]
