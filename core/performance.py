"""Performance utilities: numpy/scipy backend.

This module provides:
- numpy/scipy based array operations (CPU)
- Caching with memoization
- Optimized linear algebra via scipy/numpy
- Memory-efficient batch processing
"""

import numpy as np
import functools
import hashlib
import threading
from typing import Callable, Any, Optional, Tuple, List, Union
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

try:
    import psutil

    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False
    psutil = None


# =============================================================================
# Caching
# =============================================================================


def mesh_hash(vertices: np.ndarray, faces: np.ndarray) -> str:
    """Generate unique hash for mesh vertices and faces (numpy arrays)."""
    v_bytes = np.asarray(vertices).tobytes()
    f_bytes = np.asarray(faces).tobytes()
    return hashlib.md5(v_bytes + f_bytes).hexdigest()


def lru_cache(maxsize: int = 128):
    """LRU cache decorator for mesh-dependent functions."""

    def decorator(func: Callable) -> Callable:
        cache = {}
        cache_order = []
        lock = threading.Lock()

        @functools.wraps(func)
        def wrapper(mesh, *args, **kwargs):
            if not hasattr(mesh, "vertices") or not hasattr(mesh, "faces"):
                return func(mesh, *args, **kwargs)

            key = (mesh_hash(mesh.vertices, mesh.faces), args, tuple(sorted(kwargs.items())))

            with lock:
                if key in cache:
                    cache_order.remove(key)
                    cache_order.append(key)
                    return cache[key]

            result = func(mesh, *args, **kwargs)

            with lock:
                if key not in cache:
                    if len(cache) >= maxsize:
                        old_key = cache_order.pop(0)
                        del cache[old_key]
                    cache[key] = result
                    cache_order.append(key)

            return result

        wrapper.cache_clear = lambda: (cache.clear(), cache_order.clear())
        wrapper.cache_info = lambda: {"size": len(cache), "maxsize": maxsize}
        return wrapper

    return decorator


def cached_property(func: Callable) -> property:
    """Cached property decorator."""
    attr_name = f"_cached_{func.__name__}"
    lock = threading.Lock()

    @property
    @functools.wraps(func)
    def wrapper(self):
        if not hasattr(self, attr_name):
            with lock:
                if not hasattr(self, attr_name):
                    result = func(self)
                    object.__setattr__(self, attr_name, result)
        return getattr(self, attr_name)

    return wrapper


# =============================================================================
# Optimized Linear Algebra
# =============================================================================


def cotangent_weights(vertices: np.ndarray, faces: np.ndarray) -> dict:
    """Compute cotangent weights for all edges.

    Args:
        vertices: (n_vertices, 3) numpy array
        faces: (n_faces, 3) numpy array

    Returns:
        Dictionary of edge index -> cotangent weight. For each face the
        cotangent at a vertex is assigned to its OPPOSITE edge.
    """
    weights = {}
    for f in faces:
        i0, i1, i2 = int(f[0]), int(f[1]), int(f[2])
        v0, v1, v2 = vertices[i0], vertices[i1], vertices[i2]

        def cot(a, b, c):
            ba = b - a
            ca = c - a
            cross = np.cross(ba, ca)
            norm = np.linalg.norm(cross)
            if norm < 1e-12:
                return 0.0
            return float(np.dot(ba, ca) / norm)

        cot0 = cot(v0, v1, v2)
        cot1 = cot(v1, v2, v0)
        cot2 = cot(v2, v0, v1)

        e01 = (min(i0, i1), max(i0, i1))
        e12 = (min(i1, i2), max(i1, i2))
        e20 = (min(i2, i0), max(i2, i0))

        # Cotangent at a vertex is assigned to the OPPOSITE edge.
        weights[e12] = weights.get(e12, 0.0) + cot0
        weights[e20] = weights.get(e20, 0.0) + cot1
        weights[e01] = weights.get(e01, 0.0) + cot2

    return weights


def face_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute face areas.

    Args:
        vertices: (n_vertices, 3) numpy array
        faces: (n_faces, 3) numpy array

    Returns:
        (n_faces,) array of areas
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    return 0.5 * np.linalg.norm(cross, axis=1)


def vertex_normals(
    vertices: np.ndarray, faces: np.ndarray, normalize: bool = True
) -> np.ndarray:
    """Compute vertex normals.

    Args:
        vertices: (n_vertices, 3) numpy array
        faces: (n_faces, 3) numpy array
        normalize: Whether to normalize the normals

    Returns:
        (n_vertices, 3) array of normals
    """
    n_v = vertices.shape[0]
    normals = np.zeros((n_v, 3))

    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    for k in range(3):
        np.add.at(normals, faces[:, k], n)

    if normalize:
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms = np.where(norms > 1e-14, norms, 1.0)
        normals = normals / norms

    return normals


def vertex_areas(
    vertices: np.ndarray, faces: np.ndarray, method: str = "barycentric"
) -> np.ndarray:
    """Compute vertex areas (barycentric or voronoi).

    Args:
        vertices: (n_vertices, 3) numpy array
        faces: (n_faces, 3) numpy array
        method: 'barycentric' or 'voronoi'

    Returns:
        (n_vertices,) array of areas
    """
    n_v = vertices.shape[0]
    areas = np.zeros(n_v)
    face_areas_val = face_areas(vertices, faces)

    if method == "barycentric":
        for k in range(3):
            np.add.at(areas, faces[:, k], face_areas_val / 3.0)
    else:
        # Mixed Voronoi area (Meyer et al. 2003).
        for f, fa in zip(faces, face_areas_val):
            i0, i1, i2 = int(f[0]), int(f[1]), int(f[2])
            v0, v1, v2 = vertices[i0], vertices[i1], vertices[i2]

            def cot(a, b, c):
                ba = b - a
                ca = c - a
                cross = np.cross(ba, ca)
                norm = np.linalg.norm(cross)
                if norm < 1e-12:
                    return 0.0
                return float(np.dot(ba, ca) / norm)

            cot0 = cot(v0, v1, v2)
            cot1 = cot(v1, v2, v0)
            cot2 = cot(v2, v0, v1)

            l0 = float(np.sum((v1 - v2) ** 2))
            l1 = float(np.sum((v2 - v0) ** 2))
            l2 = float(np.sum((v0 - v1) ** 2))

            # Angles at each vertex.
            ang0 = np.arccos(np.clip(cot0 / np.sqrt(1.0 + cot0 * cot0), -1.0, 1.0)) if cot0 != 0 else np.pi / 2
            ang1 = np.arccos(np.clip(cot1 / np.sqrt(1.0 + cot1 * cot1), -1.0, 1.0)) if cot1 != 0 else np.pi / 2
            ang2 = np.arccos(np.clip(cot2 / np.sqrt(1.0 + cot2 * cot2), -1.0, 1.0)) if cot2 != 0 else np.pi / 2

            obtuse_k = None
            if ang0 > np.pi / 2:
                obtuse_k = 0
            elif ang1 > np.pi / 2:
                obtuse_k = 1
            elif ang2 > np.pi / 2:
                obtuse_k = 2

            if obtuse_k is not None:
                contrib = np.array([fa / 4.0, fa / 4.0, fa / 4.0])
                contrib[obtuse_k] = fa / 2.0
            else:
                contrib = np.array(
                    [
                        0.125 * (cot1 + cot2) * l0,
                        0.125 * (cot0 + cot2) * l1,
                        0.125 * (cot0 + cot1) * l2,
                    ]
                )
            np.add.at(areas, np.array([i0, i1, i2]), contrib)

    return np.maximum(areas, 1e-12)


# =============================================================================
# Parallel Processing
# =============================================================================


def parallel_map(
    func: Callable, items: list, n_jobs: int = -1, device: Optional[str] = None
) -> list:
    """Parallel map backed by a thread pool."""
    if n_jobs == -1:
        n_jobs = mp.cpu_count()

    if n_jobs == 1:
        return [func(item) for item in items]

    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        return list(executor.map(func, items))


# =============================================================================
# Benchmarking
# =============================================================================


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self, name: str = "Operation", verbose: bool = True):
        self.name = name
        self.verbose = verbose
        self.elapsed = 0

    def __enter__(self):
        import time

        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        import time

        self.elapsed = time.perf_counter() - self._start
        if self.verbose:
            print(f"{self.name}: {self.elapsed:.4f}s")


def benchmark(func: Callable, *args, n_runs: int = 10, warmup: int = 2, **kwargs) -> dict:
    """Benchmark a function."""
    import time

    # Warmup
    for _ in range(warmup):
        func(*args, **kwargs)

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        "mean": np.mean(times),
        "std": np.std(times),
        "min": np.min(times),
        "max": np.max(times),
        "times": times,
    }


# =============================================================================
# System Info
# =============================================================================


def get_system_info() -> dict:
    """Get system information."""
    info = {
        "n_cpus": mp.cpu_count(),
    }

    if _HAS_PSUTIL:
        mem = psutil.virtual_memory()
        info["ram_gb"] = mem.total / (1024**3)
        info["available_ram_gb"] = mem.available / (1024**3)

    try:
        import numpy

        info["numpy_version"] = numpy.__version__
    except Exception:
        pass

    try:
        import scipy

        info["scipy_version"] = scipy.__version__
    except Exception:
        pass

    return info


__all__ = [
    "mesh_hash",
    "lru_cache",
    "cached_property",
    "cotangent_weights",
    "face_areas",
    "vertex_normals",
    "vertex_areas",
    "parallel_map",
    "Timer",
    "benchmark",
    "get_system_info",
]
