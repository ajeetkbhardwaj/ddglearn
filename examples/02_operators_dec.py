"""
Example 02 — Discrete Exterior Calculus Operators
==================================================
The fundamental DEC building blocks: exterior derivatives d0, d1
and Hodge stars *0, *1, *2.  Plus the DEC Laplacian L0.

Modules used:
    ddglearn.operators.exterior_derivative — d0, d1
    ddglearn.operators.hodge_star          — hodge_star_0, hodge_star_1, hodge_star_2
    ddglearn.operators.laplacian           — laplacian_0
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from ddglearn.core.halfedge import HalfEdgeMesh
from ddglearn.core.mesh_io import load_mesh
from ddglearn.operators.exterior_derivative import d0, d1
from ddglearn.operators.hodge_star import hodge_star_0, hodge_star_1, hodge_star_2
from ddglearn.operators.laplacian import laplacian_0

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
vertices, faces = load_mesh(os.path.join(DATA, "bunny.obj"))
mesh = HalfEdgeMesh(vertices, faces)

# ---------------------------------------------------------------------------
# 1. Exterior derivatives
# ---------------------------------------------------------------------------
D0 = d0(mesh)
D1 = d1(mesh)

print("d0 (vertices → edges):", D0.shape)
print("d1 (edges   → faces):", D1.shape)

# d0 maps a vertex 0-form to edge 1-form values
vertex_ones = np.ones(mesh.n_vertices)
edge_values = D0 @ vertex_ones
print(f"\nD0 @ 1-vector  →  edge values: shape={edge_values.shape}, all ≈ 0? {np.allclose(edge_values, 0)}")

# d1 maps edge 1-form to face 2-form values
face_values = D1 @ edge_values
print(f"D1 @ edge_values  →  face values: shape={face_values.shape}, all ≈ 0? {np.allclose(face_values, 0)}")

# Fundamental identity: d1 @ d0 == 0 (boundary of boundary is empty)
print(f"\nd1 @ d0 == 0?  {np.allclose((D1 @ D0).data, 0) if hasattr(D1 @ D0, 'data') else np.allclose(D1 @ D0, 0)}")

# ---------------------------------------------------------------------------
# 2. Hodge stars (diagonal matrices)
# ---------------------------------------------------------------------------
H0 = hodge_star_0(mesh)
H1 = hodge_star_1(mesh)
H2 = hodge_star_2(mesh)

print(f"\n*0 (vertex areas):   diag shape={H0.shape}, sum={H0.diagonal().sum():.4f}")
print(f"*1 (edge ratios):   diag shape={H1.shape}, sum={H1.diagonal().sum():.4f}")
print(f"*2 (face areas):    diag shape={H2.shape}, sum={H2.diagonal().sum():.4f}")

# ---------------------------------------------------------------------------
# 3. Weak Laplacian L_c = d0^T *1 d0  (symmetric negative semi-definite)
# ---------------------------------------------------------------------------
from ddglearn.operators.hodge_star import hodge_star_0

L = laplacian_0(mesh)
print(f"\nWeak Laplacian L_c: shape={L.shape}, nnz={L.nnz}")

# Symmetry check: L_c should be symmetric (it's A^T B A)
diff = L - L.T
sym_err = abs(diff).max() if hasattr(abs(diff), 'max') else np.abs(diff.toarray()).max() if hasattr(diff, 'toarray') else float('nan')
print(f"Symmetry error (max |L_c - L_c^T|): {sym_err:.2e}")

# Negative semi-definite: smallest eigenvalue should be ≈ 0, rest negative
from scipy.sparse.linalg import eigsh
vals, _ = eigsh(L, k=5, sigma=0, which="LM")
print(f"Near-zero eigenvalues: {vals}")
print(f"All non-positive? {np.all(vals <= 1e-10)}")

# Strong Laplacian: Delta_strong = M^{-1} L_c  (NOT symmetric!)
# div(grad(u)) = -Delta_strong @ u  =  -M^{-1} L_c @ u
H0 = hodge_star_0(mesh)
print(f"\nMass matrix M: sum={H0.diagonal().sum():.4f}")

print("\nDone.")
