# DDGlearn

Discrete Differential Geometry Learning Library for Python.

DDGlearn is a Python library for discrete differential geometry, built around triangle meshes and discrete exterior calculus (DEC). It provides vectorized implementations of curvature operators, PDE solvers, spectral geometry, and shape analysis — all operating on half-edge mesh representations.

## Features

- **DEC Operators** — Hodge stars, exterior derivatives, Laplacians, gradient, divergence, curl
- **Curvature** — Gaussian, mean, principal curvatures, shape operator tensors
- **PDE Solvers** — Poisson, heat diffusion, wave equation, geodesics (Heat Method), fluids, cloth
- **Spectral Geometry** — Eigenpairs, HKS, WKS, Chebyshev filters, functional maps
- **Shape Analysis** — Optimal transport, parameterization, mesh decimation
- **Parallelism** — Thread/process pools for batched independent computations

## Quick Start

```bash
conda create -n ddg python=3.11 -y
conda activate ddg
pip install numpy scipy meshplot
pip install -e .   # or: PYTHONPATH=. pytest
```

```python
from ddglearn.core.halfedge import HalfEdgeMesh
from ddglearn.geometry.curvature import gaussian_curvature, mean_curvature
from ddglearn.spectral.hks import compute_hks

mesh = HalfEdgeMesh(*load_mesh("data/bunny.obj"))
K = gaussian_curvature(mesh)          # (n,) angle-deficit
H = mean_curvature(mesh)              # (n,) from cotangent Laplacian
t, hks = compute_hks(mesh, k=30)     # heat kernel signature
```

## Documentation

- [Getting Started](guide/getting-started.md) — Installation, first example
- [Core Meshes](guide/core.md) — Half-edge data structure, I/O, validation
- [DEC Operators](guide/operators.md) — Discrete exterior calculus operators
- [Geometry](guide/geometry.md) — Curvature, shape operators, parameterization
- [PDE Solvers](guide/pde.md) — Poisson, heat, wave, geodesics, fluids, cloth
- [Spectral](guide/spectral.md) — Eigenpairs, HKS, WKS, Chebyshev, functional maps
- [Parallel](guide/parallel.md) — CPU parallel processing
- [Visualization](guide/visualization.md) — HTML animation output
- [API Reference](api/core.md) — Complete function-level reference
- [Examples](examples.md) — Runnable example scripts
