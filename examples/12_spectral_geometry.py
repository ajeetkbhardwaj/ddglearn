"""
Example 12 — Spectral Geometry
================================
Laplacian eigendecomposition, Chebyshev filters, Heat Kernel
Signature (HKS), Wave Kernel Signature (WKS), and functional maps.

Modules used:
    ddglearn.spectral.eigen           — eigen_decomposition
    ddglearn.spectral.chebyshev       — scaled_laplacian, chebyshev_filter
    ddglearn.spectral.hks             — compute_hks
    ddglearn.spectral.wks             — compute_wks
    ddglearn.spectral.functional_map  — compute_functional_map,
                                         point_to_point_from_functional_map
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from ddglearn.core.halfedge import HalfEdgeMesh
from ddglearn.core.mesh_io import load_mesh
from ddglearn.spectral.eigen import eigen_decomposition
from ddglearn.spectral.chebyshev import scaled_laplacian, chebyshev_filter
from ddglearn.spectral.hks import compute_hks
from ddglearn.spectral.wks import compute_wks
from ddglearn.spectral.functional_map import (
    compute_functional_map,
    point_to_point_from_functional_map,
)

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
vertices, faces = load_mesh(os.path.join(DATA, "bunny.obj"))
mesh = HalfEdgeMesh(vertices, faces)

# ---------------------------------------------------------------------------
# 1. Eigendecomposition of the Laplacian
# ---------------------------------------------------------------------------
k = 30
evals, evecs = eigen_decomposition(mesh, k=k)
print(f"Laplacian eigenpairs (k={k}):")
print(f"  Eigenvalues: λ_0={evals[0]:.6f}, λ_1={evals[1]:.6f}, λ_{k-1}={evals[-1]:.6f}")
print(f"  Eigenvectors: shape={evecs.shape}")
print(f"  λ_0 ≈ 0? {evals[0] < 1e-6}")

# ---------------------------------------------------------------------------
# 2. Chebyshev spectral filter (no full eigendecomposition needed)
# ---------------------------------------------------------------------------
L_scaled = scaled_laplacian(mesh)
x = np.random.randn(mesh.n_vertices)
x /= np.linalg.norm(x)

order = 10
T_list = chebyshev_filter(mesh, x, order=order)
print(f"\nChebyshev filter (order={order}):")
print(f"  Returned {len(T_list)} polynomial evaluations")
print(f"  T_0 shape: {T_list[0].shape}, T_{order} shape: {T_list[order].shape}")

# ---------------------------------------------------------------------------
# 3. Heat Kernel Signature (HKS)
# ---------------------------------------------------------------------------
times, hks = compute_hks(mesh, k=50)
print(f"\nHKS:")
print(f"  times: shape={times.shape}, range=[{times.min():.4f}, {times.max():.4f}]")
print(f"  hks:   shape={hks.shape}")
print(f"  All non-negative? {(hks >= 0).all()}")

# ---------------------------------------------------------------------------
# 4. Wave Kernel Signature (WKS)
# ---------------------------------------------------------------------------
energies, wks = compute_wks(mesh, k=50)
print(f"\nWKS:")
print(f"  energies: shape={energies.shape}")
print(f"  wks:      shape={wks.shape}")
print(f"  All non-negative? {(wks >= 0).all()}")

# ---------------------------------------------------------------------------
# 5. Functional map between two shapes
# ---------------------------------------------------------------------------
vertices2, faces2 = load_mesh(os.path.join(DATA, "humanoid_tri.obj"))
vertices2 = vertices2 / (vertices2.max(axis=0) - vertices2.min(axis=0)).max()
mesh2 = HalfEdgeMesh(vertices2, faces2)

# Use HKS as descriptors for both shapes
_, desc1 = compute_hks(mesh, k=30)
_, desc2 = compute_hks(mesh2, k=30)

C, evecs1, evecs2 = compute_functional_map(mesh, mesh2, desc1, desc2, k=15)
print(f"\nFunctional map:")
print(f"  C matrix: shape={C.shape}")
print(f"  evecs1:   shape={evecs1.shape}")
print(f"  evecs2:   shape={evecs2.shape}")

# Point-to-point correspondence
corr = point_to_point_from_functional_map(C, evecs1, evecs2)
print(f"  Point correspondence: shape={corr.shape}")
print(f"  Mapping range: [0, {corr.max()}] (mesh2 has {mesh2.n_vertices} vertices)")

print("\nDone.")
