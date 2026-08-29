"""
Example 07 — Geometry Processing
=================================
Mesh decimation (simplification), harmonic parameterization
(flattening to 2D), and optimal transport (Wasserstein distance).

Modules used:
    ddglearn.geometry.decimation         — decimate_mesh
    ddglearn.geometry.parameterization   — harmonic_parameterization
    ddglearn.geometry.optimal_transport  — sinkhorn_wasserstein
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from ddglearn.core.halfedge import HalfEdgeMesh
from ddglearn.core.mesh_io import load_mesh
from ddglearn.geometry.decimation import decimate_mesh
from ddglearn.geometry.parameterization import harmonic_parameterization
from ddglearn.geometry.optimal_transport import sinkhorn_wasserstein

DATA = os.path.join(os.path.dirname(__file__), "..", "data")

# ---------------------------------------------------------------------------
# 1. Mesh decimation (grid-based vertex clustering)
# ---------------------------------------------------------------------------
vertices, faces = load_mesh(os.path.join(DATA, "teapot.obj"))
mesh = HalfEdgeMesh(vertices, faces)
print(f"Original teapot: {mesh.n_vertices}V, {mesh.n_faces}F")

# Coarsen by increasing grid resolution
coarse = decimate_mesh(mesh, grid_resolution=0.15)
print(f"Decimated:        {coarse.n_vertices}V, {coarse.n_faces}F")
print(f"  Reduction: {100*(1 - coarse.n_faces/mesh.n_faces):.0f}%")

# Even coarser
coarser = decimate_mesh(mesh, grid_resolution=0.3)
print(f"Coarser:          {coarser.n_vertices}V, {coarser.n_faces}F")

# ---------------------------------------------------------------------------
# 2. Harmonic parameterization (disk → 2D)
# ---------------------------------------------------------------------------
# Build a simple triangle with a hole (disk topology)
V_tri = np.array([[0,0,0],[1,0,0],[0.5,0.866,0],[0.25,0.433,0],[0.75,0.433,0]], dtype=float)
F_tri = np.array([[0,1,4],[0,4,3],[1,2,4],[2,3,4]])  # square-ish with 1 interior vertex
tri_mesh = HalfEdgeMesh(V_tri, F_tri)
print(f"\nSmall disk mesh: {tri_mesh.n_vertices}V, {tri_mesh.n_faces}F")

try:
    uv = harmonic_parameterization(tri_mesh)
    print(f"Harmonic parameterization: shape={uv.shape}")
    print(f"  U range: [{uv[:, 0].min():.4f}, {uv[:, 0].max():.4f}]")
    print(f"  V range: [{uv[:, 1].min():.4f}, {uv[:, 1].max():.4f}]")
except Exception as e:
    print(f"Parameterization failed: {e}")

# ---------------------------------------------------------------------------
# 3. Optimal transport (Sinkhorn-Wasserstein distance)
# ---------------------------------------------------------------------------
vertices_b, faces_b = load_mesh(os.path.join(DATA, "bunny.obj"))
vertices_b = vertices_b / (vertices_b.max(axis=0) - vertices_b.min(axis=0)).max()
bunny = HalfEdgeMesh(vertices_b, faces_b)
print(f"\nBunny: {bunny.n_vertices}V, {bunny.n_faces}F")

# Distribution p: uniform
n = bunny.n_vertices
p = np.ones(n) / n

# Distribution q: concentrated near the top of the mesh
z = vertices_b[:, 2]
q = np.exp(3.0 * z)
q /= q.sum()

dist, u, v = sinkhorn_wasserstein(bunny, p, q, t=1e-3, max_iter=100)
print(f"\nSinkhorn-Wasserstein distance (uniform vs top-heavy): {dist:.6f}")
print(f"  Dual u: [{u.min():.4f}, {u.max():.4f}]")
print(f"  Dual v: [{v.min():.4f}, {v.max():.4f}]")

print("\nDone.")
