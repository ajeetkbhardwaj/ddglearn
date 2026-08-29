# DDGlearn Project Guide

## Project Structure

```
ddglearn/
├── __init__.py              # Top-level package, re-exports all public API
├── core/
│   ├── __init__.py          # Core exports: HalfEdgeMesh, I/O
│   ├── halfedge.py          # HalfEdge data structure + mesh operations
│   ├── mesh_io.py           # Load/save: OBJ, OFF, PLY
│   ├── meshvalid.py         # Triangle mesh validation
│   └── performance.py       # Numpy/scipy utilities, caching, linear algebra, benchmarking
├── parallel/
│   ├── __init__.py          # Parallel exports
│   └── pool.py              # CPU parallel processing (threads + processes)
├── geometry/
│   ├── __init__.py          # Geometry exports
│   ├── curvature.py         # Gaussian, mean, principal curvatures
│   ├── decimation.py        # Grid-based mesh decimation
│   ├── optimal_transport.py # Convolutional Wasserstein distances
│   ├── parameterization.py  # Harmonic UV parameterization
│   └── shape_operator.py    # Discrete shape operator + principal directions
├── operators/
│   ├── __init__.py          # Operator exports
│   ├── cotangent_laplacian.py  # Cotangent Laplace-Beltrami
│   ├── connection.py        # Connection Laplacian + parallel transport
│   ├── curl.py              # Curl operators
│   ├── divergence.py        # Divergence operator
│   ├── exterior_derivative.py # d0, d1 incidence matrices
│   ├── gradient.py          # Gradient operator (vertex -> edge)
│   ├── hodge_star.py        # Hodge star 0, 1, 2
│   └── laplacian.py         # DEC Laplacian for 0-forms
├── pde/
│   ├── __init__.py          # PDE exports
│   ├── cloth.py             # Mass-spring cloth simulation
│   ├── fluids.py            # Eulerian fluid simulation on surfaces
│   ├── geodesics.py         # Heat method geodesic distance
│   ├── heat.py              # Implicit heat equation solver
│   ├── hodge_decomposition.py # Hodge decomposition
│   ├── poisson.py           # Poisson solver
│   └── wave.py              # Wave equation time-stepping
└── spectral/
    ├── __init__.py          # Spectral exports
    ├── chebyshev.py         # Chebyshev polynomial spectral filters
    ├── eigen.py             # Eigen decomposition of Laplacian
    ├── functional_map.py    # Functional maps for shape correspondence
    ├── hks.py               # Heat Kernel Signature
    └── wks.py               # Wave Kernel Signature
```

## Package Naming

The package was restructured from `ddg` to `ddglearn`. The top-level `__init__.py` uses `sys.path.insert` to allow intra-package imports like `from core.performance import ...` rather than `from ddglearn.core.performance import ...`.

---

## Parallelization

### Module: `ddglearn/parallel/`

**When to use:** Run N independent heavy computations concurrently across CPU cores.

| Function                                                      | Description                                                                |
| ------------------------------------------------------------- | -------------------------------------------------------------------------- |
| `parallel_map(func, items, n_jobs=-1, use_processes=False)` | Map`func` over `items` using thread or process pool                    |
| `thread_map(func, items, n_jobs=-1)`                        | Convenience wrapper for thread-based parallelism                           |
| `process_map(func, items, n_jobs=-1, chunksize=1)`          | Process-based parallelism (true CPU parallelism)                           |
| `parallel_for(func, items, n_jobs=-1, use_processes=False)` | Fire-and-forget parallel map (side-effects only)                           |
| `ParallelExecutor(n_jobs, use_processes)`                   | Reusable pool context manager with`.map()`, `.submit()`, `.gather()` |
| `get_cpu_count()`                                           | Number of logical CPUs                                                     |
| `available_memory_gb()`                                     | Available system RAM (requires psutil)                                     |

**Key constraint:** `process_map` and `ProcessPoolExecutor` require picklable callables (top-level functions, no lambdas/closures).

### Where parallelization is used in the project:

| Location                          | Function                  | What                                                                                                             |
| --------------------------------- | ------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `ddglearn/pde/geodesics.py:194` | `heat_method_geodesics` | Multi-source geodesic: each source runs independent heat solve + Poisson solve. Parallelized via`process_map`. |

### When to parallelize vs. vectorize:

- **Vectorize (numpy):** "Apply same operation to every element of a big array" — numpy internally uses BLAS/LAPACK which saturate all cores.
- **Parallelize (parallel module):** "Run N independent heavy computations" — each item involves its own substantial work (solves, file I/O, etc.).
- **Neither:** Sequential processes (time-stepping, iterative solvers where each step depends on the previous).

---

## Vectorization

### Where vectorization is applied in the project:

| File                                 | Function                   | What changed                                                                                                                     |
| ------------------------------------ | -------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `core/performance.py`              | `cotangent_weights`      | Python`for f in faces` loop building dict → Vectorized `_per_face_cotangents` helper + `np.unique`/`np.add.at`          |
| `core/performance.py`              | `vertex_areas` (voronoi) | Python`for f, fa in zip(faces)` per-face cot/angle → Vectorized cot, angles, obtuse detection, `np.add.at`                  |
| `core/halfedge.py`                 | `vertex_area_voronoi`    | Python`for idx, f in enumerate(F)` per-face cot+area with boundary → Vectorized cot, boundary mask, `np.add.at`             |
| `operators/connection.py`          | `compute_vertex_bases`   | Python`for i in range(n_v)` per-vertex basis → Broadcasting: `np.where` ref selection, vectorized projection                |
| `operators/connection.py`          | `connection_laplacian`   | Python`for (i,j),w in weights_dict.items()` per-edge rotation → Vectorized Rodrigues over all edges, bulk sparse COO assembly |
| `operators/exterior_derivative.py` | `d0`                     | Python`for ei, (u,v) in enumerate(edges)` building rows/cols → `np.repeat`/`np.tile`                                      |
| `operators/exterior_derivative.py` | `d1`                     | Python`for fi, face in enumerate(mesh.faces)` with dict lookup → Hash-based `np.searchsorted`, vectorized sign              |
| `geometry/shape_operator.py`       | `compute_vertex_normals` | Python`for f in F` per-face normal accumulation → `np.cross` over all faces + `np.add.at`                                 |
| `geometry/shape_operator.py`       | `shape_operator_tensor`  | Python`for i in range(n_v)` with `vertex_neighbors` → Undirected edge accumulation + `np.einsum` tangent projection       |
| `spectral/hks.py`                  | `compute_hks`            | Python`for ti, t in enumerate(times)` per-time exp/sum → `np.outer(times, vals)` + single matrix multiply                   |
| `spectral/wks.py`                  | `compute_wks`            | Python`for i, e in enumerate(times)` per-energy → Vectorized diff/weights matrix + single matrix multiply                     |
| `pde/cloth.py`                     | `cloth_simulation_step`  | Python`for (u,v) in edges` per-edge spring force → Vectorized edge arrays + `np.add.at`                                     |

### Common vectorization patterns used:

1. **`np.add.at(array, indices, values)`** — scatter-accumulate without overwriting (used for face-to-vertex, edge-to-vertex accumulation)
2. **`np.cross` + `np.linalg.norm`** over all faces at once (cotangents, areas, normals)
3. **`np.unique(..., return_inverse=True)` + `np.add.at`** — group-and-sum for edge-keyed dictionaries
4. **`np.outer` / broadcasting** — replace per-element loops with matrix operations (HKS, WKS time loops)
5. **`np.einsum`** — batched per-vertex matrix operations (shape operator tangent projection)
6. **Hash-based edge lookup** via `min*n_v + max` key + `np.searchsorted` — replaces Python dict lookups in `d1`
7. **`np.where` for conditional branching** — replaces per-element if/else (obtuse detection, boundary handling)

### Files NOT vectorized (and why):

| File                                                                                 | Reason                                                     |
| ------------------------------------------------------------------------------------ | ---------------------------------------------------------- |
| `heat.py`, `poisson.py` dirichlet BC loops                                       | Typically < 10 items, not worth vectorizing                |
| `wave.py`, `optimal_transport.py`                                                | Sequential solve loops, each iteration depends on previous |
| `chebyshev.py`, `eigen.py`                                                       | Small sequential loops / matrix ops (already efficient)    |
| `mesh_io.py`, `meshvalid.py`, `halfedge.py` `_build_halfedges`               | I/O parsing / structural construction, not compute-heavy   |
| `gradient.py`, `divergence.py`, `curl.py`, `laplacian.py`, `hodge_star.py` | Already vectorized numpy matrix operations                 |
| `decimation.py`, `parameterization.py`, `functional_map.py`                    | Already vectorized numpy ops                               |

---

## Key Design Decisions

1. **`shape_operator_tensor`** now returns an `(n_vertices, 3, 3)` array instead of a list of `(3, 3)` matrices. This is compatible with `principal_directions` which uses `enumerate(tensors)`, and is more efficient for batch operations. The edge-based vectorized implementation also fixes a latent boundary-vertex bug in the old `vertex_neighbors` dependency.
2. **`connection_laplacian`** was converted from a per-edge Python loop (with Rodrigues rotation) to fully vectorized rotation computation over all edges simultaneously, with bulk sparse matrix assembly via COO format.
3. **`parallel` module** uses `os.cpu_count()` for default worker count. `process_map` is for true CPU-bound parallelism (bypasses GIL via multiprocessing). `thread_map` is for I/O-bound or GIL-releasing (numpy BLAS) workloads.


### Warning 

what is the difference between vectorization and parallization and similararity and comparision most of time just answer

+ Thought: 360ms
  Vectorization: One operation applied to many data elements at once (e.g., A + B adds all elements in one CPU instruction). Data-parallel, single thread.
  Parallelization: Multiple independent operations run simultaneously on different CPU cores (e.g., 4 meshes processed at the same time). Task-parallel, multiple threads/processes.
  Similarity: Both reduce wall-clock time by doing more work per unit time.
  Key difference:
  Vectorization	Parallelization
  Splits	Data	Tasks
  Parallelism	Within one instruction (SIMD)	Across cores (threads/processes)
  GIL	Released by numpy/BLAS	Needs ProcessPoolExecutor for CPU-bound
  Overhead	Near zero	Process spawn + serialization
  When it helps	Same operation on big arrays	Many independent heavy computations
  Rule of thumb: Vectorize first (free speedup from numpy). Parallelize only when you have independent heavy tasks that can't be collapsed into a single array operation.
  ▣  Build · Big Pickle · 15.8s
  does vectorization => then parallization applied to it provide any performance over others ?
+ Thought: 9.2s
  Usually no — for a single vectorized operation, numpy's BLAS/LAPACK already uses all cores internally. Adding Python-level multiprocessing just adds overhead (serialization, process spawn) on top of what's already parallel.
  When it does help: batched independent operations. Example: computing geodesics from 100 sources — each source runs a full vectorized pipeline (heat solve → gradient → Poisson). Here parallelization gives ~linear speedup because each pipeline is independent and heavy enough to justify the overhead.
  When it doesn't: a single large matrix multiply A @ B — BLAS already uses all 10 cores. Splitting it into 10 chunks across processes adds serialization cost that exceeds any gain.
  The math:
  Scenario	Vectorize	Parallelize
  Single big array op (e.g., A @ B)	~Nx speedup (BLAS)	~0 or negative (overhead)
  N independent heavy pipelines	~Nx per pipeline	~Nx across pipelines
  Small arrays / few items	Marginal	Negative (overhead)
  Rule: Vectorize first. Then parallelize only over batches of independent vectorized pipelines — never over a single array operation.
