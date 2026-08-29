# Tutorial: Mesh Parameterization for Texture Mapping

**Problem:** To apply a 2D texture to a 3D surface, you need UV coordinates — a mapping from each 3D vertex to a 2D point. For a surface with a boundary (like a cut-open bunny), we want a mapping that minimizes distortion.

**Solution:** Harmonic parameterization solves Laplace's equation with boundary pinned to a circle, producing a smooth disk-shaped UV map.

## Step 1: Load a mesh with boundary

Harmonic parameterization requires a **disk-topology** mesh (one boundary component). The teapot has boundary edges:

```python
from ddglearn.core.mesh_io import load_mesh
from ddglearn.core.halfedge import HalfEdgeMesh

vertices, faces = load_mesh("data/teapot.obj")
mesh = HalfEdgeMesh(vertices, faces)

boundary = mesh.boundary_edges()
print(f"Vertices: {mesh.n_vertices}, Faces: {mesh.n_faces}")
print(f"Boundary edges: {len(boundary)}")
```

## Step 2: Compute harmonic parameterization

```python
from ddglearn.geometry.parameterization import harmonic_parameterization

uv = harmonic_parameterization(mesh)
print(f"UV shape: {uv.shape}")  # (n_verts, 2)
print(f"UV range: [{uv.min():.3f}, {uv.max():.3f}]")
```

## Step 3: Visualize the UV layout

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.scatter(uv[:, 0], uv[:, 1], s=1, c="steelblue")
ax.set_aspect("equal")
ax.set_title("Harmonic UV Parameterization")
ax.set_xlabel("u")
ax.set_ylabel("v")
plt.tight_layout()
plt.savefig("tutorial_uv_layout.png", dpi=150)
print("Saved tutorial_uv_layout.png")
```

## Step 4: Apply a checkerboard texture

A checkerboard pattern reveals distortion — squares should stay square if the mapping is conformal:

```python
def checkerboard(uv):
    """Generate checkerboard colors from UV coordinates."""
    scale = 5.0
    u, v = uv[:, 0] * scale, uv[:, 1] * scale
    return ((u.astype(int) + v.astype(int)) % 2).astype(float)

colors = checkerboard(uv)

from ddglearn.core.visualization import save_mesh_snapshot
save_mesh_snapshot(
    vertices, faces, colors,
    filename="tutorial_uv_checkerboard.html",
    title="Checkerboard texture via harmonic parameterization",
    colormap="gray",
)
```

```bash
open tutorial_uv_checkerboard.html
```

## Step 5: Analyze distortion

Harmonic parameterization is conformal (angle-preserving) but not isometric (distance-preserving). Let's measure the distortion:

```python
import numpy as np

# Compute edge lengths in 3D and UV
edges, _ = mesh.edge_list  # canonical directed edges
edge_3d = np.linalg.norm(
    vertices[np.array([e[0] for e in edges])]
    - vertices[np.array([e[1] for e in edges])],
    axis=1
)
edge_uv = np.linalg.norm(
    uv[np.array([e[0] for e in edges])]
    - uv[np.array([e[1] for e in edges])],
    axis=1
)

# Distortion ratio
ratio = edge_uv / (edge_3d + 1e-12)
print(f"Edge length ratio UV/3D: min={ratio.min():.3f}, max={ratio.max():.3f}, mean={ratio.mean():.3f}")
print(f"Distortion: {ratio.max()/ratio.min():.1f}x range")
```

## When to use this

| Use case | Parameterization method |
|---|---|
| Texture mapping (disk topology) | Harmonic (this tutorial) |
| Texture mapping (closed surface) | Tutte / LSCM after cutting |
| Mesh deformation | Harmonic coordinates |
| Remeshing | Delaunay in UV space |

## Complete script

```python
"""Mesh parameterization tutorial — harmonic UV mapping."""
import numpy as np
from ddglearn.core.mesh_io import load_mesh
from ddglearn.core.halfedge import HalfEdgeMesh
from ddglearn.geometry.parameterization import harmonic_parameterization
from ddglearn.core.visualization import save_mesh_snapshot

vertices, faces = load_mesh("data/teapot.obj")
mesh = HalfEdgeMesh(vertices, faces)

uv = harmonic_parameterization(mesh)

# Checkerboard texture
scale = 5.0
colors = ((uv[:, 0].astype(int) * scale + uv[:, 1].astype(int) * scale) % 2).astype(float)

save_mesh_snapshot(vertices, faces, colors,
    filename="tutorial_uv_checkerboard.html",
    title="Harmonic parameterization", colormap="gray")

# Distortion analysis
edges, _ = mesh.edge_list
e3d = np.linalg.norm(vertices[[e[0] for e in edges]] - vertices[[e[1] for e in edges]], axis=1)
euv = np.linalg.norm(uv[[e[0] for e in edges]] - uv[[e[1] for e in edges]], axis=1)
ratio = euv / (e3d + 1e-12)
print(f"Distortion ratio: {ratio.max()/ratio.min():.1f}x")
```
