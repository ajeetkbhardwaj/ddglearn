# Contributing to DDGlearn

Thanks for your interest in contributing! This document covers the development setup, coding conventions, and the workflow for submitting changes.

## Development Setup

### 1. Fork and clone

```bash
git clone https://github.com/<your-username>/discrete-differential-geometry.git
cd discrete-differential-geometry
```

### 2. Create an environment

```bash
conda create -n ddg python=3.11 -y
conda activate ddg
```

or with `venv`:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install in editable mode with dev tools

```bash
pip install -e ".[dev,viz]"
```

This installs NumPy, SciPy, pytest, psutil, and the visualization stack (matplotlib, meshplot, polyscope).

### 4. Verify the setup

```bash
pytest            # should pass ~58 tests in under a second
```

## Project Layout

```
ddglearn/
├── core/         # HalfEdgeMesh, mesh I/O (OBJ/OFF/PLY), validation, perf utilities
├── operators/    # DEC: d0/d1, Hodge stars, Laplacians, gradient/divergence/curl, connection
├── geometry/     # Curvatures, shape operator, optimal transport, parameterization, decimation
├── pde/          # Poisson, heat, wave, geodesics (Heat Method), fluids, cloth, Hodge decomposition
├── spectral/     # Eigenpairs, HKS/WKS, Chebyshev filters, functional maps
└── parallel/     # Thread/process pools for batched independent computations
data/             # Sample meshes (bunny, teapot, ...)
docs/             # MkDocs documentation (guide/, tutorials/, api/)
examples/         # Runnable example scripts, one per topic
tests/            # Pytest suite
```

## Coding Conventions

### Imports

- **Always use package-relative imports inside `ddglearn/`:**
  ```python
  # good — within ddglearn/pde/
  from ..operators.gradient import gradient
  from .poisson import solve_poisson

  # bad — breaks installed packages
  from operators.gradient import gradient
  ```
- Never add `sys.path` hacks to library code. Test/example scripts may insert the repo root only.

### Numerical conventions

These are project-wide invariants — do not change them without updating docs and tests together:

| Convention | Rule |
|---|---|
| Weak Laplacian | `L = d0ᵀ *1 d0`, symmetric PSD. Strong form is `-M⁻¹L`. |
| PDE solves | Mass matrix stays on the RHS (`Lu = Mf`); never pre-multiply by `M⁻¹`. |
| Heat solves | `(H0 + tW) u_next = H0 u` — **plus** sign; W is a stiffness matrix. |
| Hodge star \*1 | Cotangent weights clamped non-negative → keeps L PSD. |
| Wave scheme | Crank–Nicolson implicit; must remain unconditionally stable. |
| Edge lists | Canonical orientation: `(i, j)` with `i < j`. |

### Performance

- Vectorize with NumPy before considering threads/processes (`np.add.at`, broadcasting, `np.einsum`).
- Prefer `scipy.sparse` for matrices over ~1000 rows.
- New heavy per-element loops need justification in the PR description.

### Docstrings

Google style, with types — these are rendered into the API reference via mkdocstrings:

```python
def geodesic_distance(mesh, source_indices, time_scale=1e-3, method="poisson") -> np.ndarray:
    """Compute geodesic distance from source vertices to all vertices.

    Args:
        mesh: The half-edge mesh.
        source_indices: Vertex index or list of indices.
        time_scale: Heat diffusion time; smaller = sharper.
        method: "poisson", "varadhan", or "fmm_graph".

    Returns:
        (n_vertices,) array of distances.
    """
```

Annotate parameters/returns where practical — griffe warnings appear in CI otherwise.

## Testing

Run the full suite:

```bash
pytest
```

Requirements for new contributions:

- **New functions need tests.** Put them in the matching `tests/test_*.py` module.
- Validate against analytic solutions where possible (spheres, tori, planes) — see `tests/test_numerical.py` for patterns.
- Tests must run in seconds; avoid large meshes or long time-stepping loops.

## Documentation

Docs are MkDocs + mkdocstrings. After changing docstrings or adding features:

```bash
pip install mkdocs mkdocs-material "mkdocstrings[python]"
mkdocs serve      # preview at http://127.0.0.1:8000
```

- Public API pages use `:::` directives in `docs/api/*.md` — add new functions there.
- User-facing workflows belong in `docs/guide/` or `docs/tutorials/`.
- Tutorials should solve a concrete problem with meshes from `data/`.

## Submitting Changes

1. Create a branch: `git checkout -b feature/my-change`
2. Make your changes; add/update tests and docs.
3. Run the checks locally:
   ```bash
   pytest
   mkdocs build
   pip install build twine && python -m build && twine check dist/*
   ```
4. Push and open a Pull Request against `main`.

CI runs on every PR:

| Job | What it does |
|---|---|
| `test` | pytest matrix: Ubuntu × Python 3.9–3.12, plus macOS & Windows smoke tests |
| `build` | Builds sdist/wheel, `twine check`, installs wheel in a clean venv and imports it |
| `docs` | Builds the MkDocs site |

All three must pass before merge.

## Reporting Issues

Include:

- Python version and OS
- Minimal reproduction script (synthetic meshes preferred)
- Full traceback
- Expected vs actual behavior

For numerical surprises (unexpected negatives, NaNs, instability), include mesh stats: vertex count, boundary edge count, connected components (`mesh.boundary_edges()`, etc.).

## License

By contributing you agree that your contributions are licensed under the MIT License.
