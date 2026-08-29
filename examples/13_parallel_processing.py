"""
Example 13 — Parallel Processing
==================================
CPU-parallel map, process map, and parallel-for patterns
for accelerating per-vertex or per-mesh computations.

Modules used:
    ddglearn.parallel.pool — parallel_map, process_map, thread_map,
                              parallel_for, ParallelExecutor,
                              get_cpu_count, available_memory_gb
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import numpy as np
from ddglearn.core.halfedge import HalfEdgeMesh
from ddglearn.core.mesh_io import load_mesh
from ddglearn.geometry.curvature import gaussian_curvature, mean_curvature
from ddglearn.parallel.pool import (
    parallel_map,
    process_map,
    thread_map,
    parallel_for,
    ParallelExecutor,
    get_cpu_count,
    available_memory_gb,
)

DATA = os.path.join(os.path.dirname(__file__), "..", "data")

# ---------------------------------------------------------------------------
# 1. System info
# ---------------------------------------------------------------------------
print(f"CPU count:      {get_cpu_count()}")
mem = available_memory_gb()
print(f"Available RAM:  {mem:.1f} GB" if mem else "Available RAM:  (psutil not installed)")

# ---------------------------------------------------------------------------
# 2. Load multiple meshes (simulate a batch)
# ---------------------------------------------------------------------------
mesh_files = ["bunny.obj", "teapot.obj", "humanoid_tri.obj"]
meshes = []
for fname in mesh_files:
    path = os.path.join(DATA, fname)
    v, f = load_mesh(path)
    meshes.append(HalfEdgeMesh(v, f))
    print(f"  {fname}: {v.shape[0]}V, {f.shape[0]}F")

# ---------------------------------------------------------------------------
# 3. Serial vs parallel Gaussian curvature
# ---------------------------------------------------------------------------
def compute_K(mesh):
    """Top-level function for process_map (must be picklable)."""
    from ddglearn.geometry.curvature import gaussian_curvature
    return gaussian_curvature(mesh)

# Serial
t0 = time.perf_counter()
K_serial = [gaussian_curvature(m) for m in meshes]
t_serial = time.perf_counter() - t0
print(f"\nSerial Gaussian curvature:   {t_serial:.4f}s")

# Parallel (threaded — good because numpy releases the GIL)
t0 = time.perf_counter()
K_thread = parallel_map(compute_K, meshes, n_jobs=-1, use_processes=False)
t_thread = time.perf_counter() - t0
print(f"Threaded Gaussian curvature: {t_thread:.4f}s")

# Parallel (multiprocessing — true CPU parallelism)
t0 = time.perf_counter()
K_proc = process_map(compute_K, meshes, n_jobs=-1)
t_proc = time.perf_counter() - t0
print(f"Process Gaussian curvature:  {t_proc:.4f}s")

# Verify results match
for i, (ks, kp) in enumerate(zip(K_serial, K_proc)):
    print(f"  mesh {i}: serial vs process diff = {np.abs(ks - kp).max():.2e}")

# ---------------------------------------------------------------------------
# 4. Thread map (convenience wrapper)
# ---------------------------------------------------------------------------
t0 = time.perf_counter()
K_tm = thread_map(compute_K, meshes, n_jobs=2)
t_tm = time.perf_counter() - t0
print(f"\nthread_map (2 workers): {t_tm:.4f}s")

# ---------------------------------------------------------------------------
# 5. parallel_for (fire-and-forget, side effects only)
# ---------------------------------------------------------------------------
results = [None] * len(meshes)

def compute_and_store(i):
    from ddglearn.geometry.curvature import mean_curvature
    results[i] = mean_curvature(meshes[i])

parallel_for(compute_and_store, list(range(len(meshes))), n_jobs=2, use_processes=True)
print(f"\nparallel_for results: {[r.shape for r in results if r is not None]}")

# ---------------------------------------------------------------------------
# 6. ParallelExecutor (reusable pool with submit/gather)
# ---------------------------------------------------------------------------
with ParallelExecutor(n_jobs=2, use_processes=False) as pool:
    futures = [pool.submit(gaussian_curvature, m) for m in meshes]
    K_exec = pool.gather(futures)

print(f"\nParallelExecutor: {len(K_exec)} results gathered")
for i, k in enumerate(K_exec):
    print(f"  mesh {i}: shape={k.shape}")

print("\nDone.")
