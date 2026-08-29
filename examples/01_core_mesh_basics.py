"""
Example 01 — Core Mesh Basics
==============================
Mesh I/O, half-edge construction, validation, and basic queries.

Modules used:
    ddglearn.core.halfedge     — HalfEdgeMesh
    ddglearn.core.mesh_io      — load_mesh, save_mesh
    ddglearn.core.meshvalid    — validate_triangle_mesh
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from ddglearn.core.halfedge import HalfEdgeMesh
from ddglearn.core.mesh_io import load_mesh, save_mesh
from ddglearn.core.meshvalid import validate_triangle_mesh

# ---------------------------------------------------------------------------
# 1. Load a mesh from disk
# ---------------------------------------------------------------------------
DATA = os.path.join(os.path.dirname(__file__), "..", "data")

vertices, faces = load_mesh(os.path.join(DATA, "bunny.obj"))
print(f"Loaded bunny.obj: {vertices.shape[0]} vertices, {faces.shape[0]} faces")

# Build the half-edge data structure
mesh = HalfEdgeMesh(vertices, faces)
print(f"  n_vertices={mesh.n_vertices}, n_faces={mesh.n_faces}, n_edges={mesh.n_edges}")

# ---------------------------------------------------------------------------
# 2. Construct a mesh from scratch (a regular tetrahedron)
# ---------------------------------------------------------------------------
V_tet = np.array([
    [ 1.0,  1.0,  1.0],
    [ 1.0, -1.0, -1.0],
    [-1.0,  1.0, -1.0],
    [-1.0, -1.0,  1.0],
], dtype=float)

F_tet = np.array([
    [0, 1, 2],
    [0, 1, 3],
    [0, 2, 3],
    [1, 2, 3],
])

tet = HalfEdgeMesh(V_tet, F_tet)
print(f"\nTetrahedron: {tet.n_vertices}V, {tet.n_faces}F, {tet.n_edges}E")

# ---------------------------------------------------------------------------
# 3. Mesh validation
# ---------------------------------------------------------------------------
validate_triangle_mesh(vertices, faces)
print("bunny.obj passed validation (non-degenerate, index-bounds OK, no NaNs)")

# ---------------------------------------------------------------------------
# 4. Basic topology queries
# ---------------------------------------------------------------------------
v = 0
neighbors = mesh.vertex_neighbors(v)
print(f"\nVertex {v} neighbors: {neighbors}")

boundary = mesh.boundary_edges()
print(f"Boundary edges: {len(boundary)}")

# ---------------------------------------------------------------------------
# 5. Vertex areas
# ---------------------------------------------------------------------------
va_bary = mesh.vertex_area_barycentric()
va_voro = mesh.vertex_area_voronoi()
print(f"\nVertex areas (barycentric): sum={va_bary.sum():.6f}")
print(f"Vertex areas (Voronoi):     sum={va_voro.sum():.6f}")

# ---------------------------------------------------------------------------
# 6. Save mesh back to disk
# ---------------------------------------------------------------------------
out_path = os.path.join(DATA, "bunny_copy.obj")
save_mesh(out_path, vertices, faces)
print(f"\nSaved copy to {out_path}")

# ---------------------------------------------------------------------------
# 7. Edge list and edge map
# ---------------------------------------------------------------------------
edge_list = mesh.edge_list
edge_map = mesh.edge_map
print(f"\nEdge list length: {len(edge_list)}")
print(f"Edge map entries: {len(edge_map)}")
print(f"First 5 edges: {edge_list[:5]}")

print("\nDone.")
