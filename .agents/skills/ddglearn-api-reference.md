# DDGlearn — Full Public API Reference

**Package:** `ddglearn` v0.1.0
**Python:** 3.11+ | **Dependencies:** numpy, scipy, meshplot (optional)

---

## 1. `ddglearn.core` — Core Mesh Data Structures and I/O

### 1.1 `core/halfedge.py`

| Symbol | Kind | Signature | Description |
|---|---|---|---|
| `HalfEdge` | class | `(origin: int, face: int)` | Lightweight half-edge record. Slots: `origin`, `face`, `twin`, `next`, `prev`. |
| `HalfEdgeMesh` | class | `(vertices: np.ndarray, faces: np.ndarray, validate_manifold: bool = True)` | Robust half-edge mesh for triangle meshes. Stores `vertices (n,3)` and `faces (m,3)`. Builds half-edge connectivity, validates manifold property. |

**HalfEdgeMesh methods:**

| Method | Returns | Description |
|---|---|---|
| `n_halfedges` | `int` | Number of half-edges (property). |
| `n_edges` | `int` | Number of unique edges (cached property). |
| `n_vertices` | `int` | Number of vertices (property). |
| `edge_list` | `List[Tuple[int,int]]` | Canonical directed edge pairs (cached). |
| `edge_map` | `dict` | `{(min,max): index}` edge key mapping (cached). |
| `outgoing_halfedge(v)` | `Optional[int]` | Outgoing half-edge from vertex `v`. |
| `vertex_neighbors(v)` | `List[int]` | 1-ring neighbor vertex indices. |
| `boundary_edges()` | `List[Tuple[int,int]]` | Boundary edge pairs (half-edges with no twin). |
| `vertex_area_barycentric()` | `np.ndarray` `(n,)` | Barycentric vertex areas. |
| `vertex_area_voronoi()` | `np.ndarray` `(n,)` | Mixed Voronoi area per vertex (Meyer et al. 2002). |

---

### 1.2 `core/mesh_io.py`

| Function | Signature | Description |
|---|---|---|
| `load_mesh` | `(filepath: str, clean: bool = True) -> Tuple[ndarray, ndarray]` | Auto-load OBJ/OFF/PLY by extension. If `clean=True`, removes duplicate vertices. |
| `save_mesh` | `(filepath: str, vertices, faces) -> None` | Auto-save to OBJ/OFF/PLY by extension. |
| `load_obj` | `(filepath: str) -> Tuple[ndarray, ndarray]` | Load OBJ file. Handles `v`, `v/t`, `v/t/n` face formats. |
| `save_obj` | `(filepath: str, vertices, faces) -> None` | Save OBJ file. |
| `load_off` | `(filepath: str) -> Tuple[ndarray, ndarray]` | Load OFF file. Triangulates polygonal faces. |
| `save_off` | `(filepath: str, vertices, faces) -> None` | Save OFF file. |
| `load_ply` | `(filepath: str) -> Tuple[ndarray, ndarray]` | Load ASCII PLY file. |
| `save_ply` | `(filepath: str, vertices, faces) -> None` | Save ASCII PLY file. |
| `remove_duplicate_vertices` | `(vertices, faces, tol=1e-8) -> Tuple[ndarray, ndarray]` | Vectorized spatial hashing dedup (O(n log n)). Removes degenerate faces post-dedup. |

---

### 1.3 `core/meshvalid.py`

| Function | Signature | Description |
|---|---|---|
| `validate_triangle_mesh` | `(vertices, faces) -> None` | Validates index bounds, NaN vertices, degenerate triangles. Raises `ValueError`. |

---

### 1.4 `core/performance.py`

| Symbol | Kind | Signature | Description |
|---|---|---|---|
| `mesh_hash` | function | `(vertices, faces) -> str` | MD5 hash of mesh data for cache keys. |
| `lru_cache` | decorator | `(maxsize=128)` | Thread-safe LRU cache for mesh-dependent functions. Keyed on mesh hash + args. |
| `cached_property` | decorator | `(func) -> property` | Thread-safe cached property. |
| `cotangent_weights` | function | `(vertices, faces) -> dict` | Cotangent weights for all edges. Returns `{(i,j): weight}`. |
| `face_areas` | function | `(vertices, faces) -> ndarray` `(m,)` | Per-face areas via cross product. |
| `vertex_normals` | function | `(vertices, faces, normalize=True) -> ndarray` `(n,3)` | Area-weighted vertex normals. |
| `vertex_areas` | function | `(vertices, faces, method="barycentric") -> ndarray` `(n,)` | Vertex areas. `method`: `"barycentric"` or `"voronoi"`. |
| `Timer` | class | `(name="Operation", verbose=True)` | Context manager. `.elapsed` holds duration. |
| `benchmark` | function | `(func, *args, n_runs=10, warmup=2, **kwargs) -> dict` | Returns `{mean, std, min, max, times}`. |
| `get_system_info` | function | `() -> dict` | CPU count, RAM, numpy/scipy versions. |

---

### 1.5 `core/visualization.py`

| Function | Signature | Description |
|---|---|---|
| `save_mesh_animation` | `(vertices, faces, frames, filename="animation.html", title="Simulation", colormap="coolwarm", vmin=None, vmax=None, fps=10)` | Save vertex-color animation as standalone HTML with slider+playback. `frames`: `(n_frames, n_verts)`. |
| `save_mesh_snapshot` | `(vertices, faces, colors, filename="snapshot.html", title="", colormap="coolwarm", vmin=None, vmax=None)` | Save single colored mesh as standalone HTML. |

---

## 2. `ddglearn.geometry` — Curvature, Shape Operators, Transport, Parameterization

### 2.1 `geometry/curvature.py`

| Function | Signature | Description |
|---|---|---|
| `gaussian_curvature` | `(mesh) -> ndarray` `(n,)` | Angle-deficit Gaussian curvature per vertex. Boundary: `pi - sum`, interior: `2pi - sum`, divided by Voronoi area. |
| `mean_curvature_vector` | `(mesh) -> ndarray` `(n,3)` | Discrete mean curvature normal `Hn` via cotangent Laplacian: `L @ V / A`. |
| `mean_curvature` | `(mesh) -> ndarray` `(n,)` | Scalar mean curvature: `0.5 * \|\|Hn\|\|`. |
| `principal_curvatures` | `(mesh) -> (k1, k2)` | Principal curvatures from `K` and `H`: `k1 = H + sqrt(H^2 - K)`, `k2 = H - sqrt(H^2 - K)`. |

---

### 2.2 `geometry/shape_operator.py`

| Function | Signature | Description |
|---|---|---|
| `compute_vertex_normals` | `(mesh) -> ndarray` `(n,3)` | Area-weighted vertex normals. |
| `shape_operator_tensor` | `(mesh) -> ndarray` `(n,3,3)` | Symmetric 3x3 curvature tensor per vertex via edge-based cotangent weights projected to tangent plane. |
| `principal_directions` | `(mesh) -> (k1_dirs, k2_dirs)` | Principal curvature directions via eigendecomposition of shape operator tensor. Each `(n,3)`. |

---

### 2.3 `geometry/optimal_transport.py`

| Function | Signature | Description |
|---|---|---|
| `sinkhorn_wasserstein` | `(mesh, p, q, t=1e-3, max_iter=150, tol=1e-5) -> (dist, u, v)` | Convolutional Wasserstein distance (Solomon et al. 2015). Heat-regularized Sinkhorn in O(N). `p`, `q`: `(n,)` distributions. |

---

### 2.4 `geometry/parameterization.py`

| Function | Signature | Description |
|---|---|---|
| `harmonic_parameterization` | `(mesh) -> ndarray` `(n,2)` | 2D harmonic parameterization for disk-topology meshes. Boundary mapped to unit circle. |

---

### 2.5 `geometry/decimation.py`

| Function | Signature | Description |
|---|---|---|
| `decimate_mesh` | `(mesh, grid_resolution=0.05) -> HalfEdgeMesh` | Grid-based vertex clustering decimation. Larger `grid_resolution` = fewer polygons. |

---

## 3. `ddglearn.operators` — Discrete Exterior Calculus Operators

### 3.1 `operators/exterior_derivative.py`

| Function | Signature | Description |
|---|---|---|
| `edge_list_and_map` | `(mesh) -> (List[Tuple[int,int]], dict)` | Canonical directed edge list + `{(min,max): index}` map. Cached on mesh. |
| `d0` | `(mesh) -> sparse/dense` `(n_edges, n_vertices)` | Exterior derivative d0: vertices → edges. `-1` at source, `+1` at target. |
| `d1` | `(mesh) -> sparse/dense` `(n_faces, n_edges)` | Exterior derivative d1: edges → faces. `+1`/`-1` based on orientation. |

---

### 3.2 `operators/hodge_star.py`

| Function | Signature | Description |
|---|---|---|
| `hodge_star_0` | `(mesh) -> sparse/dense` `(n_v, n_v)` | Hodge star \*0: diagonal matrix of Voronoi vertex areas. |
| `hodge_star_1` | `(mesh) -> sparse/dense` `(n_e, n_e)` | Hodge star \*1: diagonal matrix of cotangent weights per edge (clamped ≥ 0). |
| `hodge_star_2` | `(mesh) -> sparse/dense` `(n_f, n_f)` | Hodge star \*2: diagonal matrix of face areas. |

---

### 3.3 `operators/laplacian.py`

| Function | Signature | Description |
|---|---|---|
| `laplacian_0` | `(mesh) -> sparse/dense` `(n_v, n_v)` | Weak Laplacian (Stiffness Matrix): `L = d0^T *1 d0`. Symmetric PSD. |

**Convention:** The strong Laplacian (pointwise) is `Δ_strong = -M^{-1} L_c`. Never use `M^{-1} L_c` directly in linear solves — keep the mass matrix on the RHS: `L_c x = M b`.

---

### 3.4 `operators/cotangent_laplacian.py`

| Function | Signature | Description |
|---|---|---|
| `cotangent_laplacian` | `(mesh) -> sparse/dense` `(n_v, n_v)` | Cotangent Laplace-Beltrami. Off-diagonal ≤ 0, diagonal is positive sum. |

---

### 3.5 `operators/gradient.py`

| Function | Signature | Description |
|---|---|---|
| `gradient` | `(mesh, u) -> ndarray` `(n_e,)` | Gradient of scalar field `u` on edges: `d0 @ u`. |
| `gradient_vector` | `(mesh, u) -> ndarray` `(n_f, 3)` | 3D gradient vector on each face. |

---

### 3.6 `operators/divergence.py`

| Function | Signature | Description |
|---|---|---|
| `divergence` | `(mesh, v) -> ndarray` `(n_v,)` | Divergence of 1-form `v` (edge values). `div v = -*0^{-1} d0^T *1 v`. |
| `divergence_face_vector` | `(mesh, X) -> ndarray` `(n_v,)` | Divergence of constant-per-face vector field `X` `(n_f, 3)`. |

---

### 3.7 `operators/curl.py`

| Function | Signature | Description |
|---|---|---|
| `curl_scalar` | `(mesh, u) -> ndarray` `(n_v,)` | Laplace-Beltrami of scalar field `u`. |
| `curl_vector` | `(mesh, v) -> ndarray` `(n_f,)` | Curl of 1-form `v`: maps edge values to face values via `d1 @ v`. |

---

### 3.8 `operators/connection.py`

| Function | Signature | Description |
|---|---|---|
| `compute_vertex_bases` | `(mesh) -> ndarray` `(n, 2, 3)` | Orthogonal tangent basis `(X, Y)` for each vertex. |
| `connection_laplacian` | `(mesh) -> (L_conn, bases)` | Connection Laplacian for tangent vector fields via Rodrigues rotation. `L_conn`: `(2n, 2n)` sparse. |
| `vertex_holonomy` | `(mesh) -> ndarray` `(n,)` | Holonomy (angle defect) around each vertex. Equals `K * A` (Gauss-Bonnet). |

---

## 4. `ddglearn.pde` — PDE Solvers

### 4.1 `pde/poisson.py`

| Function | Signature | Description |
|---|---|---|
| `solve_poisson` | `(mesh, f, pin_index=0, pin_value=0.0, dirichlet_indices=None, dirichlet_values=None, neumann_indices=None, neumann_values=None) -> ndarray` `(n,)` | Solve `W u = H0 f` where `W = d0^T *1 d0`. Supports pinning, Dirichlet, Neumann BCs. |

---

### 4.2 `pde/heat.py`

| Function | Signature | Description |
|---|---|---|
| `implicit_heat_step` | `(mesh, u, t, pin_index=0, dirichlet_indices=None, dirichlet_values=None) -> ndarray` `(n,)` | Single implicit Euler heat step. Solves `(H0 + t W) u_next = H0 u`. |
| `implicit_heat` | `(mesh, u0, t, steps=1, pin_index=0, ...) -> ndarray` `(n,)` | Multi-step implicit heat diffusion. |

---

### 4.3 `pde/wave.py`

| Function | Signature | Description |
|---|---|---|
| `wave_step` | `(mesh, u_prev, u_curr, dt, pin_index=0) -> ndarray` `(n,)` | Single wave step (Crank-Nicolson / Newmark-beta). Solves `(H0 + dt^2/2 W) u_next = 2 H0 u_curr - (H0 + dt^2/2 W) u_prev`. Unconditionally stable, 2nd-order. |
| `simulate_wave` | `(mesh, u0, dt, steps, u1=None, pin_index=0) -> ndarray` `(n,)` | Multi-step wave simulation. |

---

### 4.4 `pde/geodesics.py`

| Function | Signature | Description |
|---|---|---|
| `solve_heat_diffusion` | `(mesh, source_indices, time_scale=1e-3, ...) -> ndarray` `(n,)` | Solve `(I + t L) u = delta_source` for heat kernel. |
| `geodesic_distance` | `(mesh, source_indices, time_scale=1e-3, method="poisson") -> ndarray` `(n,)` | Geodesic distance via Heat Method (Crane et al. 2013). Methods: `"poisson"`, `"varadhan"`, `"fmm_graph"`. |
| `heat_method_geodesics` | `(mesh, source_indices=None, time_scale=1e-3, batch=False, method="poisson") -> ndarray` | Batch geodesics. `batch=True`: `(n,)` nearest source. `batch=False`: `(n_sources, n)`. |

---

### 4.5 `pde/hodge_decomposition.py`

| Function | Signature | Description |
|---|---|---|
| `hodge_decomposition` | `(mesh, v_edges) -> (grad_f, div_free, harmonic)` | Decompose 1-form into curl-free + div-free + harmonic. `v_edges`: `(n_e,)`. |
| `is_divergence_free` | `(mesh, v_edges, tol=1e-6) -> bool` | Check if 1-form is divergence-free. |
| `is_curl_free` | `(mesh, v_edges, tol=1e-6) -> bool` | Check if 1-form is curl-free. |

---

### 4.6 `pde/fluids.py`

| Function | Signature | Description |
|---|---|---|
| `fluid_velocity_from_vorticity` | `(mesh, omega, viscosity=0.0, dt=0.01) -> (flux, omega_diffused)` | Incompressible fluid velocity from surface vorticity. Solves Poisson for stream function, takes gradient. Returns `(n_e,)` flux, `(n_v,)` diffused vorticity. |

---

### 4.7 `pde/cloth.py`

| Function | Signature | Description |
|---|---|---|
| `cloth_simulation_step` | `(mesh, pos, vel, dt=0.01, mass=1.0, stiffness=1000.0, damping=0.99, gravity=-9.81, pinned_vertices=None) -> (new_pos, new_vel)` | Semi-implicit mass-spring cloth step. Spring forces along edges, gravity, damping. Both `(n_v, 3)`. |

---

## 5. `ddglearn.spectral` — Spectral Geometry

### 5.1 `spectral/eigen.py`

| Function | Signature | Description |
|---|---|---|
| `eigen_decomposition` | `(mesh, k=20) -> (eigenvalues, eigenvectors)` | First `k` eigenpairs of 0-form Laplacian. Returns `(k,)` and `(n_v, k)`. Uses scipy `eigsh` or dense `eigh`. |

---

### 5.2 `spectral/hks.py`

| Function | Signature | Description |
|---|---|---|
| `compute_hks` | `(mesh, k=50, times=None) -> (times, hks)` | Heat Kernel Signature: `HKS(x,t) = sum_i exp(-lambda_i * t) * phi_i(x)^2`. Returns `(T,)` times, `(n_v, T)` signatures. |

---

### 5.3 `spectral/wks.py`

| Function | Signature | Description |
|---|---|---|
| `compute_wks` | `(mesh, k=100, times=None) -> (energies, wks)` | Wave Kernel Signature (Aubry et al. 2011). Gaussian-windowed spectral density. Returns `(T,)` log-energies, `(n_v, T)` signatures. |

---

### 5.4 `spectral/chebyshev.py`

| Function | Signature | Description |
|---|---|---|
| `scaled_laplacian` | `(mesh) -> sparse` | Scaled symmetric normalized Laplacian `L_sym = M^{-1/2} L M^{-1/2}`, rescaled to `[-1, 1]`. |
| `chebyshev_filter` | `(mesh, x, order) -> List[ndarray]` | Chebyshev polynomial filters `[T_0(L)x, ..., T_k(L)x]` for spectral graph convolutions without eigendecomposition. |

---

### 5.5 `spectral/functional_map.py`

*Must be imported explicitly: `from ddglearn.spectral.functional_map import ...`*

| Function | Signature | Description |
|---|---|---|
| `compute_functional_map` | `(mesh1, mesh2, desc1, desc2, k=30) -> (C, evecs1, evecs2)` | Functional Map matrix (Ovsjanikov et al. 2012). `desc1/2`: `(n_v, n_desc)` descriptors. Returns `C: (k,k)`, `evecs1: (n1,k)`, `evecs2: (n2,k)`. |
| `point_to_point_from_functional_map` | `(C, evecs1, evecs2) -> ndarray` `(n1,)` | Convert functional map to point-to-point correspondence via nearest-neighbor. |

---

## 6. `ddglearn.parallel` — CPU Parallel Processing

### 6.1 `parallel/pool.py`

| Symbol | Kind | Signature | Description |
|---|---|---|---|
| `get_cpu_count` | function | `() -> int` | Number of logical CPUs. |
| `available_memory_gb` | function | `() -> Optional[float]` | Available RAM in GB (requires psutil). |
| `parallel_map` | function | `(func, items, n_jobs=-1, use_processes=False) -> list` | Map `func` over items with thread or process pool. `n_jobs=-1` = all CPUs. |
| `thread_map` | function | `(func, items, n_jobs=-1) -> list` | Convenience: `parallel_map(..., use_processes=False)`. |
| `process_map` | function | `(func, items, n_jobs=-1, chunksize=1) -> list` | Process-based map. `func` must be picklable (defined at module level). |
| `parallel_for` | function | `(func, items, n_jobs=-1, use_processes=False) -> None` | Fire-and-forget map (discards returns). For side-effecting work. |
| `ParallelExecutor` | class | `(n_jobs=-1, use_processes=False)` | Reusable pool context manager. Methods: `map(func, items)`, `submit(func, *args)`, `gather(futures)`. |

---

## Cross-Module Dependency Graph

```
ddglearn
  |
  +-- core/halfedge          --> numpy, core.meshvalid
  +-- core/mesh_io           --> numpy
  +-- core/performance       --> numpy, psutil(opt)
  +-- core/visualization     --> numpy, meshplot(lazy)
  |
  +-- geometry/curvature     --> operators.cotangent_laplacian(lazy)
  +-- geometry/shape_operator--> numpy
  +-- geometry/optimal_transport --> operators.{d0, hodge_star}(lazy), scipy(opt)
  +-- geometry/parameterization  --> pde.poisson(lazy)
  +-- geometry/decimation        --> core.halfedge(lazy)
  |
  +-- operators/exterior_derivative --> scipy.sparse(opt)
  +-- operators/hodge_star          --> .exterior_derivative
  +-- operators/laplacian           --> .exterior_derivative, .hodge_star
  +-- operators/cotangent_laplacian --> scipy.sparse(opt)
  +-- operators/gradient            --> operators.{exterior_derivative, hodge_star}
  +-- operators/divergence          --> operators.{exterior_derivative, hodge_star}
  +-- operators/curl                --> operators.{exterior_derivative, laplacian}(lazy)
  +-- operators/connection          --> core.performance(lazy), geometry.curvature(lazy)
  |
  +-- pde/poisson            --> operators.{d0, hodge_star}
  +-- pde/heat               --> operators.{d0, hodge_star}
  +-- pde/wave               --> operators.{d0, hodge_star}
  +-- pde/geodesics          --> operators.{gradient, divergence, d0, hodge_star}, pde.poisson
  +-- pde/hodge_decomposition--> operators.{gradient, divergence, curl, d0, d1, hodge_star}, pde.poisson
  +-- pde/fluids             --> operators.{gradient, hodge_star_1}, pde.{poisson, heat}
  +-- pde/cloth              --> operators.exterior_derivative(lazy)
  |
  +-- spectral/eigen         --> operators.laplacian(lazy), scipy.sparse.linalg
  +-- spectral/hks           --> spectral.eigen
  +-- spectral/wks           --> spectral.eigen
  +-- spectral/chebyshev     --> operators.{cotangent_laplacian, hodge_star_0}
  +-- spectral/functional_map--> spectral.eigen(lazy), operators.hodge_star(lazy)
  |
  +-- parallel/pool          --> concurrent.futures, psutil(opt)
```

---

## Available Meshes (`data/`)

| File | Vertices | Faces | Notes |
|---|---|---|---|
| `bunny.obj` | 3485 | 6966 | Fully connected, no boundary. Best for PDE demos. |
| `teapot.obj` | 529 | 992 | 3 disconnected components, 64 boundary edges. |
| `humanoid_tri.obj` | 64 | 96 | Small, mostly disconnected (8/64 connected). |
| `airboat.obj` | 5797 | 11566 | Degenerate triangles — fails HalfEdgeMesh construction. |
| `skyscraper.obj` | 2022 | 3692 | Degenerate triangles — fails HalfEdgeMesh construction. |
| `violin_case.obj` | 1080 | 2120 | Degenerate triangles — fails HalfEdgeMesh construction. |
