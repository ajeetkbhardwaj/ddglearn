# Getting Started

## Installation

```bash
conda create -n ddg python=3.11 -y
conda activate ddg
pip install numpy scipy meshplot
```

Install the package in development mode:

```bash
cd discrete-differential-geometry
pip install -e .
# or: PYTHONPATH=. pytest
```

## First Example

```python
from ddglearn.core.mesh_io import load_mesh
from ddglearn.core.halfedge import HalfEdgeMesh
from ddglearn.geometry.curvature import gaussian_curvature
from ddglearn.operators.laplacian import laplacian_0

vertices, faces = load_mesh("data/bunny.obj")
mesh = HalfEdgeMesh(vertices, faces)

K = gaussian_curvature(mesh)   # (n,) angle-deficit
L = laplacian_0(mesh)          # (n,n) weak Laplacian: d0^T *1 d0
```

## Conventions

| Concept | Convention |
|---|---|
| Weak Laplacian | `L = d0^T *1 d0` (symmetric PSD stiffness matrix) |
| Strong Laplacian | `Δ_strong = -M^{-1} L` (pointwise, not used in solves) |
| Poisson solve | `L u = M f` (mass matrix on RHS) |
| Hodge stars | `*0` = vertex areas, `*1` = cotangent weights (clamped ≥ 0), `*2` = face areas |
| Edge list | Canonical directed: `(i,j)` with `i < j` |

## Data Files

All sample meshes are in `data/`:

| File | Vertices | Notes |
|---|---|---|
| `bunny.obj` | 3485 | Best for PDE demos — fully connected, no boundary |
| `teapot.obj` | 529 | 3 components, 64 boundary edges |
| `humanoid_tri.obj` | 64 | Small, mostly disconnected |
