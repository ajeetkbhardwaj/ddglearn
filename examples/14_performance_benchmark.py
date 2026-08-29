"""
Example 14 — Performance Benchmarking
=======================================
Timing utilities, system info, caching, and mesh hashing
for reproducible benchmarks.

Modules used:
    ddglearn.core.performance — Timer, benchmark, get_system_info,
                                 mesh_hash, lru_cache, cached_property
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from ddglearn.core.halfedge import HalfEdgeMesh
from ddglearn.core.mesh_io import load_mesh
from ddglearn.core.performance import (
    Timer,
    benchmark,
    get_system_info,
    mesh_hash,
    lru_cache,
)

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
vertices, faces = load_mesh(os.path.join(DATA, "bunny.obj"))
mesh = HalfEdgeMesh(vertices, faces)

# ---------------------------------------------------------------------------
# 1. System info
# ---------------------------------------------------------------------------
info = get_system_info()
print("System info:")
for k, v in info.items():
    print(f"  {k}: {v}")

# ---------------------------------------------------------------------------
# 2. Mesh hashing (for deduplication / caching)
# ---------------------------------------------------------------------------
h = mesh_hash(vertices, faces)
print(f"\nMesh hash (bunny): {h}")

# Same mesh, same hash
h2 = mesh_hash(vertices, faces)
print(f"Hash stable? {h == h2}")

# Different mesh, different hash
vertices2, faces2 = load_mesh(os.path.join(DATA, "teapot.obj"))
h3 = mesh_hash(vertices2, faces2)
print(f"Different mesh hash? {h != h3}")

# ---------------------------------------------------------------------------
# 3. Timer context manager
# ---------------------------------------------------------------------------
from ddglearn.geometry.curvature import gaussian_curvature

with Timer("Gaussian curvature") as t:
    K = gaussian_curvature(mesh)
print(f"Timer recorded: {t.elapsed:.6f}s")

# ---------------------------------------------------------------------------
# 4. Benchmark function (automated multi-run timing)
# ---------------------------------------------------------------------------
result = benchmark(gaussian_curvature, mesh, n_runs=5, warmup=1)
print(f"\nBenchmark (5 runs):")
print(f"  mean:   {result['mean']:.6f}s")
print(f"  std:    {result['std']:.6f}s")
print(f"  min:    {result['min']:.6f}s")
print(f"  max:    {result['max']:.6f}s")
print(f"  times:  {[f'{t:.6f}' for t in result['times']]}")

# ---------------------------------------------------------------------------
# 5. Custom cached function with lru_cache
# ---------------------------------------------------------------------------
call_count = 0

@lru_cache(maxsize=16)
def expensive_vertex_area(mesh, method="barycentric"):
    """Compute vertex areas with caching — second call is free."""
    global call_count
    call_count += 1
    return mesh.vertex_area_voronoi()

# First call: computes
val1 = expensive_vertex_area(mesh)
print(f"\nCached function: call_count={call_count} (expect 1)")

# Same args: hits cache
val2 = expensive_vertex_area(mesh)
print(f"After cache hit: call_count={call_count} (expect 1)")
print(f"Values match: {np.array_equal(val1, val2)}")

# Different mesh: computes again
mesh2 = HalfEdgeMesh(*load_mesh(os.path.join(DATA, "teapot.obj")))
val3 = expensive_vertex_area(mesh2)
print(f"After new mesh:  call_count={call_count} (expect 2)")

# ---------------------------------------------------------------------------
# 6. Benchmark two implementations
# ---------------------------------------------------------------------------
from ddglearn.operators.cotangent_laplacian import cotangent_laplacian
from ddglearn.operators.laplacian import laplacian_0

b1 = benchmark(cotangent_laplacian, mesh, n_runs=5, warmup=1)
b2 = benchmark(laplacian_0, mesh, n_runs=5, warmup=1)

print(f"\nLaplacian benchmarks:")
print(f"  Cotangent:  {b1['mean']*1000:.2f} ms ± {b1['std']*1000:.2f} ms")
print(f"  DEC:        {b2['mean']*1000:.2f} ms ± {b2['std']*1000:.2f} ms")

print("\nDone.")
