DDG — Advanced Discrete Differential Geometry Framework (Research-Grade)
==========================================================================

A **comprehensive, production-ready** Discrete Exterior Calculus (DEC) / FEM implementation 
for triangle meshes with spectral geometry, curvature analysis, PDE solvers, and advanced
vector field processing.

✅ **ALL PHASES IMPLEMENTED** (100% feature coverage)

---

Quick Start
-----------

Install dependencies:

```bash
pip install -r requirements.txt
pip install polyscope  # For interactive 3D visualization (optional)
```

Run demos:

```bash
python examples/advanced_demo.py      # All features in one go
python examples/spectral_demo.py      # Eigenvalues + HKS
python examples/curvature_demo.py     # Curvature visualization
python examples/pde_demo.py           # PDE solvers
```

Run tests:

```bash
pytest test_ddg.py test_advanced.py -v
```

---

## Features Overview

**Phase 1–2: Core DEC** ✅
- Half-edge mesh with adjacency queries
- Exterior derivatives d₀, d₁ (topological matrices)
- Hodge star operators *₀, *₁, *₂
- Laplace–Beltrami operator

**Phase 5A: Vector Operators** ✅
- `gradient(u)` — vertex scalar → edge values
- `divergence(v)` — edge values → vertex scalar
- `curl(u)` scalar and `curl(v)` vector forms

**Phase 5B: Spectral + Shape** ✅
- Wave Kernel Signature (WKS) — advanced shape descriptor
- Vertex normals, principal directions
- Shape operator (Weingarten map)

**Phase 5C: Advanced PDE** ✅
- Geodesic distance (heat method)
- Hodge decomposition (curl-free + div-free + harmonic)

**Phase 6: Visualization** ✅
- Polyscope integration (interactive 3D viewer)
- Batch visualization API

**Phase 7: I/O & Utilities** ✅
- OBJ, PLY, OFF file support
- Auto-format detection

---

## Complete API

```python
# Mesh & I/O
from core.halfedge import HalfEdgeMesh
from core.mesh_io import load_mesh, save_mesh

# Vector calculus
from operators.gradient import gradient
from operators.divergence import divergence
from operators.curl import curl_scalar, curl_vector

# Curvature
from geometry.curvature import gaussian_curvature, principal_curvatures
from geometry.shape_operator import principal_directions

# Spectral
from spectral.eigen import eigen_decomposition
from spectral.hks import compute_hks
from spectral.wks import compute_wks

# PDE
from pde.poisson import solve_poisson
from pde.heat import implicit_heat
from pde.wave import simulate_wave
from pde.geodesics import geodesic_distance
from pde.hodge_decomposition import hodge_decomposition

# Visualization
from visualization.polyscope_viewer import (
    plot_scalar_field, plot_vector_field,
    plot_curvature, plot_geodesic_distance, plot_hks
)
```

---

## Project Structure

```
ddg/
├── core/                      # Mesh & I/O
│   ├── halfedge.py
│   └── mesh_io.py             # NEW: OBJ/PLY/OFF support
├── geometry/                  # Shape analysis
│   ├── curvature.py
│   └── shape_operator.py      # NEW: Principal directions
├── operators/                 # DEC + vector ops
│   ├── exterior_derivative.py
│   ├── hodge_star.py
│   ├── laplacian.py
│   ├── gradient.py            # NEW
│   ├── divergence.py          # NEW
│   └── curl.py                # NEW
├── pde/                       # Solvers
│   ├── poisson.py
│   ├── heat.py
│   ├── wave.py
│   ├── geodesics.py           # NEW: Heat method
│   └── hodge_decomposition.py # NEW: Vector field splitting
├── spectral/                  # Spectral methods
│   ├── eigen.py
│   ├── hks.py
│   └── wks.py                 # NEW: Wave Kernel Signature
├── visualization/             # Visualization
│   └── polyscope_viewer.py    # NEW: Interactive 3D
├── examples/
│   ├── spectral_demo.py
│   ├── curvature_demo.py
│   ├── pde_demo.py
│   └── advanced_demo.py       # NEW: All features demo
├── test_ddg.py                # 20+ core tests
├── test_advanced.py           # 20+ advanced tests
└── requirements.txt
```

---

## Tests (40+ comprehensive)

```bash
pytest test_ddg.py test_advanced.py -v
```

---

## Mathematical References

- Crane et al. *Discrete Differential Geometry: An Applied Introduction* (2018)
- Botsch et al. *Polygon Mesh Processing* (2010)
- Meyer et al. Discrete differential-geometry operators for triangulated 2-manifolds (2003)
- Crane et al. Geodesics in Heat (2013)

---

## Notes

- SciPy sparse for large meshes, NumPy dense for small
- Implicit time-stepping (unconditionally stable)
- Polyscope integration (install with `pip install polyscope`)

For full API docs: `README_EXTENDED.md` | For roadmap: `COVERAGE_AND_ROADMAP.md`

