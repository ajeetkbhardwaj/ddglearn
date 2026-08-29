"""Performance utilities: numpy/scipy backend.

This module provides:
- numpy/scipy based array operations (CPU)
- Caching with memoization
- Optimized linear algebra via scipy/numpy
- Memory-efficient batch processing
"""

import os
import numpy as np
import functools
import hashlib
import threading
from typing import Callable, Any, Optional, Tuple, List, Union

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


def _per_face_cotangents(vertices: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized cotangents of the corner angles for every face.

    Returns the cotangent of the angle at v0, v1, v2 per face as three
    arrays of shape (n_faces,). Degenerate (zero-area) corners contribute 0.
    """
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]

    def cot_vec(ba: np.ndarray, ca: np.ndarray) -> np.ndarray:
        cross = np.cross(ba, ca)
        norm = np.linalg.norm(cross, axis=1)
        # Zero-area (degenerate) corners contribute no cotangent weight.
        return np.where(norm < 1e-12, 0.0, np.sum(ba * ca, axis=1) / np.maximum(norm, 1e-12))

    cot0 = cot_vec(v1 - v0, v2 - v0)
    cot1 = cot_vec(v2 - v1, v0 - v1)
    cot2 = cot_vec(v0 - v2, v1 - v2)
    return cot0, cot1, cot2


def cotangent_weights(vertices: np.ndarray, faces: np.ndarray) -> dict:
    """Compute cotangent weights for all edges.

    Args:
        vertices: (n_vertices, 3) numpy array
        faces: (n_faces, 3) numpy array

    Returns:
        Dictionary of edge index -> cotangent weight. For each face the
        cotangent at a vertex is assigned to its OPPOSITE edge.
    """
    F = np.asarray(faces)
    cot0, cot1, cot2 = _per_face_cotangents(np.asarray(vertices, dtype=float), F)

    # Edge opposite to each vertex: (v1,v2) for cot0, (v2,v0) for cot1, ...
    ei = np.concatenate([
        np.minimum(F[:, 1], F[:, 2]),
        np.minimum(F[:, 2], F[:, 0]),
        np.minimum(F[:, 0], F[:, 1]),
    ])
    ej = np.concatenate([
        np.maximum(F[:, 1], F[:, 2]),
        np.maximum(F[:, 2], F[:, 0]),
        np.maximum(F[:, 0], F[:, 1]),
    ])
    vals = np.concatenate([cot0, cot1, cot2])

    keys = np.stack([ei, ej], axis=1)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    sums = np.zeros(len(uniq))
    np.add.at(sums, inv, vals)
    return {(int(uniq[k, 0]), int(uniq[k, 1])): float(sums[k]) for k in range(len(uniq))}


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
    F = np.asarray(faces)
    areas = np.zeros(n_v)
    face_areas_val = face_areas(vertices, faces)

    if method == "barycentric":
        for k in range(3):
            np.add.at(areas, F[:, k], face_areas_val / 3.0)
        return np.maximum(areas, 1e-12)

    # Mixed Voronoi area (Meyer et al. 2003), vectorized over faces.
    V = np.asarray(vertices, dtype=float)
    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cot0, cot1, cot2 = _per_face_cotangents(V, F)

    l0 = np.sum((v1 - v2) ** 2, axis=1)
    l1 = np.sum((v2 - v0) ** 2, axis=1)
    l2 = np.sum((v0 - v1) ** 2, axis=1)

    ang0 = np.arccos(np.clip(cot0 / np.sqrt(1.0 + cot0 * cot0), -1.0, 1.0))
    ang1 = np.arccos(np.clip(cot1 / np.sqrt(1.0 + cot1 * cot1), -1.0, 1.0))
    ang2 = np.arccos(np.clip(cot2 / np.sqrt(1.0 + cot2 * cot2), -1.0, 1.0))

    obtuse0 = ang0 > np.pi / 2
    obtuse1 = ang1 > np.pi / 2
    obtuse2 = ang2 > np.pi / 2
    any_obtuse = obtuse0 | obtuse1 | obtuse2

    c0 = np.where(obtuse0, face_areas_val / 2.0,
                  np.where(any_obtuse, face_areas_val / 4.0, 0.125 * (cot1 + cot2) * l0))
    c1 = np.where(obtuse1, face_areas_val / 2.0,
                  np.where(any_obtuse, face_areas_val / 4.0, 0.125 * (cot0 + cot2) * l1))
    c2 = np.where(obtuse2, face_areas_val / 2.0,
                  np.where(any_obtuse, face_areas_val / 4.0, 0.125 * (cot0 + cot1) * l2))

    np.add.at(areas, F[:, 0], c0)
    np.add.at(areas, F[:, 1], c1)
    np.add.at(areas, F[:, 2], c2)

    return np.maximum(areas, 1e-12)


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
        "n_cpus": os.cpu_count() or 1,
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
    "Timer",
    "benchmark",
    "get_system_info",
]
