"""
Example 05 — Curvature: Gaussian, Mean, and Principal
======================================================
Compute the three fundamental curvature measures on a surface.

Modules used:
    ddglearn.geometry.curvature — gaussian_curvature,
                                   mean_curvature_vector,
                                   mean_curvature,
                                   principal_curvatures
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from ddglearn.core.halfedge import HalfEdgeMesh
from ddglearn.core.mesh_io import load_mesh
from ddglearn.geometry.curvature import (
    gaussian_curvature,
    mean_curvature_vector,
    mean_curvature,
    principal_curvatures,
)

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
vertices, faces = load_mesh(os.path.join(DATA, "bunny.obj"))
# Normalize to unit bounding box so curvature values are scale-appropriate
vertices = vertices / (vertices.max(axis=0) - vertices.min(axis=0)).max()
mesh = HalfEdgeMesh(vertices, faces)

# ---------------------------------------------------------------------------
# 1. Gaussian curvature K (angle defect / Voronoi area)
# ---------------------------------------------------------------------------
K = gaussian_curvature(mesh)
print(f"Gaussian curvature K:")
print(f"  shape={K.shape}, min={K.min():.6f}, max={K.max():.6f}")

# Gauss-Bonnet: ∫ K dA = 2π χ(M)
from ddglearn.core.halfedge import HalfEdgeMesh
va = mesh.vertex_area_voronoi()
int_K = np.dot(K, va)
chi = mesh.n_vertices - mesh.n_edges + mesh.n_faces
print(f"  ∫K dA = {int_K:.6f}")
print(f"  2π χ   = {2*np.pi*chi:.6f}")

# ---------------------------------------------------------------------------
# 2. Mean curvature H (via Laplacian position)
# ---------------------------------------------------------------------------
Hn = mean_curvature_vector(mesh)  # (n_vertices, 3) — mean curvature normal
H = mean_curvature(mesh)          # (n_vertices,)   — scalar magnitude

print(f"\nMean curvature normal: shape={Hn.shape}")
print(f"Mean curvature H:      min={H.min():.6f}, max={H.max():.6f}")

# ---------------------------------------------------------------------------
# 3. Principal curvatures κ1, κ2
# ---------------------------------------------------------------------------
k1, k2 = principal_curvatures(mesh)
print(f"\nPrincipal curvatures:")
print(f"  κ1: min={k1.min():.6f}, max={k1.max():.6f}")
print(f"  κ2: min={k2.min():.6f}, max={k2.max():.6f}")

# Check: H = (κ1 + κ2) / 2, K = κ1 · κ2
H_from_k = 0.5 * (k1 + k2)
K_from_k = k1 * k2
print(f"\n  H vs (κ1+κ2)/2: max diff = {np.abs(H - H_from_k).max():.2e}")
print(f"  K vs κ1·κ2:      max diff = {np.abs(K - K_from_k).max():.2e}")

# ---------------------------------------------------------------------------
# 4. Curvature on a sphere (ground truth: K = 1/r², H = 1/r)
# ---------------------------------------------------------------------------
from ddglearn.core.halfedge import HalfEdgeMesh as HE

# Build a simple sphere via icosahedron subdivision
phi = (1 + np.sqrt(5)) / 2  # golden ratio
V_sphere = np.array([
    [-1,  phi, 0], [ 1,  phi, 0], [-1, -phi, 0], [ 1, -phi, 0],
    [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
    [ phi, 0, -1], [ phi, 0,  1], [-phi, 0, -1], [-phi, 0,  1],
], dtype=float)
V_sphere /= np.linalg.norm(V_sphere[0])  # normalize to unit sphere

F_sphere = np.array([
    [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
    [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
    [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
    [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1],
])

sphere = HE(V_sphere, F_sphere)
K_s = gaussian_curvature(sphere)
H_s = mean_curvature(sphere)
print(f"\nUnit sphere Gaussian curvature: mean={K_s.mean():.6f} (expect ≈ 1.0)")
print(f"Unit sphere mean curvature:     mean={H_s.mean():.6f} (expect ≈ 1.0)")

print("\nDone.")
