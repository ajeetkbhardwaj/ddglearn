"""
Example 06 — Shape Operator and Principal Directions
=====================================================
The shape operator tensor S: maps tangent vectors to tangent
vectors, encoding how the normal changes across the surface.

Modules used:
    ddglearn.geometry.shape_operator — compute_vertex_normals,
                                        shape_operator_tensor,
                                        principal_directions
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from ddglearn.core.halfedge import HalfEdgeMesh
from ddglearn.core.mesh_io import load_mesh
from ddglearn.geometry.shape_operator import (
    compute_vertex_normals,
    shape_operator_tensor,
    principal_directions,
)

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
vertices, faces = load_mesh(os.path.join(DATA, "teapot.obj"))
mesh = HalfEdgeMesh(vertices, faces)

# ---------------------------------------------------------------------------
# 1. Vertex normals
# ---------------------------------------------------------------------------
normals = compute_vertex_normals(mesh)
print(f"Vertex normals: shape={normals.shape}")
print(f"  All unit length? {np.allclose(np.linalg.norm(normals, axis=1), 1.0)}")

# ---------------------------------------------------------------------------
# 2. Shape operator tensor S
# ---------------------------------------------------------------------------
S = shape_operator_tensor(mesh)
print(f"\nShape operator tensor: shape={S.shape}")  # (n_vertices, 3, 3)

# S should be symmetric
for i in range(3):
    for j in range(3):
        asym = np.abs(S[:, i, j] - S[:, j, i]).max()
        if asym > 1e-10:
            print(f"  WARNING: S[{i},{j}] asymmetry = {asym:.2e}")

# S should annihilate normals: S @ n ≈ 0
Sn = np.einsum("vij,vj->vi", S, normals)
print(f"|S @ n| max: {np.abs(Sn).max():.2e} (should be ≈ 0)")

# ---------------------------------------------------------------------------
# 3. Principal directions and curvatures
# ---------------------------------------------------------------------------
k1_dirs, k2_dirs = principal_directions(mesh)
print(f"\nPrincipal direction k1: shape={k1_dirs.shape}")
print(f"Principal direction k2: shape={k2_dirs.shape}")

# Directions should be orthogonal to normals
dot1 = np.sum(k1_dirs * normals, axis=1)
dot2 = np.sum(k2_dirs * normals, axis=1)
print(f"k1 · n max: {np.abs(dot1).max():.2e}")
print(f"k2 · n max: {np.abs(dot2).max():.2e}")

# k1 and k2 should be orthogonal to each other
dot12 = np.sum(k1_dirs * k2_dirs, axis=1)
print(f"k1 · k2 max: {np.abs(dot12).max():.2e}")

print("\nDone.")
