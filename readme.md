# DDGlearn

**Discrete Differential Geometry Library for Python**

A production-grade implementation of Discrete Exterior Calculus (DEC) on triangle meshes — curvature analysis, PDE solvers, spectral geometry, and shape understanding, all fully vectorized with NumPy.

## Features

| Module        | What it provides                                                                                                                                     |
| ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `core`      | Half-edge mesh data structure, OBJ/OFF/PLY I/O, validation, caching & benchmarking utilities                                                         |
| `operators` | DEC operators: exterior derivatives`d0`/`d1`, Hodge stars `*0/*1/*2`, (co)tangent Laplacians, gradient, divergence, curl, connection Laplacian |
| `geometry`  | Gaussian/mean/principal curvatures, shape operator tensors, optimal transport, harmonic parameterization, mesh decimation                            |
| `pde`       | Poisson, implicit heat, unconditionally-stable wave equation, Heat-Method geodesics, Hodge decomposition, surface fluids, cloth simulation           |
| `spectral`  | Laplacian eigenpairs, Heat/Wave Kernel Signatures (HKS/WKS), Chebyshev filters, functional maps for shape correspondence                             |
| `parallel`  | Thread/process pools for batched independent computations                                                                                            |

## Installation

```bash
pip install .
```

With optional extras:

```bash
pip install ".[viz]"   # matplotlib, meshplot, polyscope
pip install ".[dev]"   # pytest, psutil, mkdocs toolchain
pip install ".[all]"
```

Requires Python ≥ 3.9, NumPy, SciPy.

## Quick Start

```python
import numpy as np
from ddglearn import HalfEdgeMesh, load_mesh

# Load a mesh (half-edge data structure)
vertices, faces = load_mesh("data/bunny.obj")
mesh = HalfEdgeMesh(vertices, faces)

# Curvature
from ddglearn import gaussian_curvature, mean_curvature
K = gaussian_curvature(mesh)      # angle-deficit per vertex
H = mean_curvature(mesh)          # via cotangent Laplacian

# Geodesic distance (Heat Method, Crane et al. 2013)
from ddglearn import geodesic_distance
d = geodesic_distance(mesh, source_indices=[0])

# Spectral descriptors
from ddglearn import compute_hks, eigen_decomposition
evals, evecs = eigen_decomposition(mesh, k=20)
times, hks = compute_hks(mesh, k=50)

# PDEs on the surface
from ddglearn import solve_poisson, implicit_heat_step, wave_step
u = solve_poisson(mesh, f, pin_index=0)          # Δu = f
u_next = implicit_heat_step(mesh, u, t=1e-5)     # heat diffusion
```

## Conventions

- **Weak Laplacian:** `L = d0ᵀ *1 d0` — symmetric positive semi-definite. Strong form is `-M⁻¹L`; PDE solves keep the mass matrix on the RHS (`Lu = Mf`).
- **Hodge star \*1** uses cotangent weights clamped to be non-negative → guarantees PSD Laplacian.
- **Wave equation** uses Crank–Nicolson (unconditionally stable, 2nd-order).
- **Geodesics** implement the Heat Method with `poisson`, `varadhan`, and `fmm_graph` variants.

## Documentation

Full documentation lives in `docs/`:

```bash
pip install mkdocs mkdocs-material mkdocstrings[python]
mkdocs serve    # http://127.0.0.1:8000
```

- [Getting Started](docs/guide/getting-started.md)
- [Guides](docs/guide/core.md) — core, operators, geometry, PDE, spectral, parallel, visualization
- [Tutorials](docs/tutorials/geodesic-distance.md) — real problems solved with the bundled meshes:
  geodesic distance · spectral fingerprinting · surface fluids · parameterization · shape correspondence
- [API Reference](docs/api/core.md) — auto-generated from docstrings via mkdocstrings

## Examples

Runnable scripts in [`examples/`](examples/) covering every submodule; PDE/spectral examples write interactive WebGL HTML animations to `examples/_output/`.

```bash
python examples/02_curvature.py
python examples/09_pde_wave.py
open examples/_output/09_wave.html
```

## Testing

```bash
pytest            # from repo root
```

## Sample Meshes

`data/` bundles several meshes (Stanford bunny, Utah teapot, humanoid, …). Note: `bunny.obj` is watertight and ideal for PDE demos; some others contain degenerate triangles or multiple components — see the docs guide for compatibility notes.

## License

MIT — see [LICENSE](LICENSE).
