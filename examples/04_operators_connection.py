"""
Example 04 — Connection, Parallel Transport, and Holonomy
==========================================================
Tangent bases, the Connection Laplacian, and vertex holonomy
(angle defect = discrete Gaussian curvature).

Modules used:
    ddglearn.operators.connection — compute_vertex_bases,
                                     connection_laplacian,
                                     vertex_holonomy
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from ddglearn.core.halfedge import HalfEdgeMesh
from ddglearn.core.mesh_io import load_mesh
from ddglearn.operators.connection import (
    compute_vertex_bases,
    connection_laplacian,
    vertex_holonomy,
)

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
vertices, faces = load_mesh(os.path.join(DATA, "bunny.obj"))
mesh = HalfEdgeMesh(vertices, faces)

# ---------------------------------------------------------------------------
# 1. Vertex tangent bases: (X, Y) per vertex
# ---------------------------------------------------------------------------
bases = compute_vertex_bases(mesh)
print(f"Vertex bases: shape={bases.shape}")  # (n_vertices, 2, 3)

# Check orthogonality: X · Y ≈ 0
dots = np.sum(bases[:, 0, :] * bases[:, 1, :], axis=1)
print(f"X·Y orthogonality: max|dot|={np.abs(dots).max():.2e}")

# Check unit length
len_x = np.linalg.norm(bases[:, 0, :], axis=1)
len_y = np.linalg.norm(bases[:, 1, :], axis=1)
print(f"|X| range: [{len_x.min():.6f}, {len_x.max():.6f}]")
print(f"|Y| range: [{len_y.min():.6f}, {len_y.max():.6f}]")

# ---------------------------------------------------------------------------
# 2. Connection Laplacian
# ---------------------------------------------------------------------------
L_conn, conn_bases = connection_laplacian(mesh)
print(f"\nConnection Laplacian: shape={L_conn.shape}")

# Symmetry check
diff = L_conn - L_conn.T
sym_err = abs(diff).max() if hasattr(abs(diff), 'max') else np.abs(diff.toarray()).max()
print(f"Symmetry error: {sym_err:.2e}")

# ---------------------------------------------------------------------------
# 3. Vertex holonomy (angle defect)
# ---------------------------------------------------------------------------
hol = vertex_holonomy(mesh)
print(f"\nVertex holonomy: min={hol.min():.6f}, max={hol.max():.6f}")

# Total holonomy = 2π χ (Gauss-Bonnet)
chi = mesh.n_vertices - mesh.n_edges + mesh.n_faces  # Euler characteristic
print(f"Total holonomy:  {hol.sum():.6f}")
print(f"2π · χ(M):       {2 * np.pi * chi:.6f}")
print(f"Match (Gauss-Bonnet)? {np.isclose(hol.sum(), 2 * np.pi * chi, rtol=0.05)}")

print("\nDone.")
