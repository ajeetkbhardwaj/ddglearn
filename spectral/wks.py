"""Wave Kernel Signature (WKS): wavelet-based shape descriptor.

WKS is similar to HKS but uses a different exponential weighting function
and is often better at capturing fine-grained shape details.

Formula: WKS_t(x) = ∑_i e^{i\sqrt{λ_i}t} |φ_i(x)|²

This is a complex-valued alternative to HKS, effectively a Fourier transform
of the spectral density function.
"""
import numpy as np
from typing import Tuple

from spectral.eigen import eigen_decomposition


def compute_wks(mesh, k: int = 50, times: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """Compute WKS for mesh using first k Laplacian eigenpairs.

    Returns (times, wks) where wks has shape (n_vertices, len(times)).
    
    Notes:
    - WKS uses oscillating exponentials (complex-valued math, real output via magnitude)
    - Better than HKS at distinguishing features at different scales
    - More sensitive to local geometry than HKS
    """
    vals, vecs = eigen_decomposition(mesh, k=k)
    n = mesh.n_vertices
    m = vecs.shape[1]

    # choose times if not provided: logspace over frequency/energy range
    if times is None:
        positive = vals[vals > 1e-12]
        if positive.size == 0:
            tmin, tmax = 1e-3, 10.0
        else:
            # WKS naturally covers frequencies ~ sqrt(λ), so scale accordingly
            tmin = 1e-2
            tmax = 4.0 * np.pi / (positive[-1] + 1e-12)
        times = np.logspace(np.log10(tmin), np.log10(tmax), num=10)

    # compute squared eigenfunctions (stationary part)
    phi2 = vecs ** 2  # shape (n, m)

    wks = np.zeros((n, times.size), dtype=float)
    for ti, t in enumerate(times):
        # oscillating part: e^{i \sqrt{λ} t}
        freq_factors = np.exp(1j * np.sqrt(np.maximum(vals, 0.0)) * t)
        # combine with amplitude |φ_i|²
        wks[:, ti] = np.abs(phi2.dot(freq_factors)).real

    return times, wks


__all__ = ["compute_wks"]
