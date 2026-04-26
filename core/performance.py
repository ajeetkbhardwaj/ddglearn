"""Performance utilities: PyTorch backend with GPU acceleration.

This module provides:
- PyTorch-based tensor operations (GPU/CPU)
- Device management (CUDA, MPS, CPU)
- Caching with GPU-aware memoization
- Automatic differentiation support
- Optimized linear algebra via torch
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
    import torch
    from torch import tensor, as_tensor, zeros, ones
    from torch.linalg import norm

    _HAS_TORCH = True
    _TORCH_VERSION = torch.__version__
except ImportError:
    _HAS_TORCH = False
    torch = None
    tensor = np.array
    as_tensor = np.array
    norm = np.linalg.norm

try:
    import psutil

    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False
    psutil = None


# =============================================================================
# Device Management
# =============================================================================


class DeviceManager:
    """Manage compute devices (CPU, CUDA, MPS)."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._device = torch.device("cpu") if not _HAS_TORCH else torch.device("cpu")
        self._preferred_dtype = torch.float64
        self._cache_dir = None
        self._initialized = True

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def has_cuda(self) -> bool:
        return _HAS_TORCH and torch.cuda.is_available()

    @property
    def has_mps(self) -> bool:
        return _HAS_TORCH and hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    @property
    def n_devices(self) -> int:
        if self.has_cuda:
            return torch.cuda.device_count()
        elif self.has_mps:
            return 1
        return 1

    def set_device(self, device: Union[str, int] = "cuda"):
        """Set the compute device.

        Args:
            device: 'cpu', 'cuda', 'cuda:0', 'mps', or device index
        """
        if not _HAS_TORCH:
            self._device = torch.device("cpu")
            return

        if isinstance(device, int):
            device = f"cuda:{device}"

        if device.startswith("cuda"):
            if torch.cuda.is_available():
                self._device = torch.device(device)
            else:
                print("CUDA not available, falling back to CPU")
                self._device = torch.device("cpu")
        elif device == "mps" and self.has_mps:
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")

    def to_device(self, data: Any) -> Any:
        """Move data to the current device."""
        if not _HAS_TORCH:
            return data

        if isinstance(data, torch.Tensor):
            return data.to(self._device)
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data).to(self._device)
        elif isinstance(data, (list, tuple)):
            return type(data)(self.to_device(x) for x in data)
        elif isinstance(data, dict):
            return {k: self.to_device(v) for k, v in data.items()}
        return data

    def to_cpu(self, data: Any) -> Any:
        """Move data to CPU (detaching from GPU graph)."""
        if not _HAS_TORCH:
            return data

        if isinstance(data, torch.Tensor):
            return data.cpu().detach()
        elif isinstance(data, (list, tuple)):
            return type(data)(self.to_cpu(x) for x in data)
        elif isinstance(data, dict):
            return {k: self.to_cpu(v) for k, v in data.items()}
        return data

    def set_dtype(self, dtype: str = "float64"):
        """Set default dtype: 'float32' or 'float64'."""
        if dtype == "float32":
            self._preferred_dtype = torch.float32
        else:
            self._preferred_dtype = torch.float64

    @property
    def dtype(self) -> torch.dtype:
        return self._preferred_dtype

    def memory_info(self) -> dict:
        """Get memory information."""
        if not self.has_cuda:
            return {"available": False}

        return {
            "available": True,
            "allocated": torch.cuda.memory_allocated() / 1024**3,
            "reserved": torch.cuda.memory_reserved() / 1024**3,
            "max_allocated": torch.cuda.max_memory_allocated() / 1024**3,
        }

    def clear_cache(self):
        """Clear CUDA cache."""
        if self.has_cuda:
            torch.cuda.empty_cache()


_device_manager = DeviceManager()


def get_device() -> torch.device:
    """Get the current compute device."""
    return _device_manager.device


def has_cuda() -> bool:
    """Check if CUDA is available."""
    return _device_manager.has_cuda


def has_mps() -> bool:
    """Check if Apple MPS is available."""
    return _device_manager.has_mps


def set_device(device: Union[str, int] = "cuda"):
    """Set the compute device."""
    _device_manager.set_device(device)


def to_device(data: Any) -> Any:
    """Move data to current device."""
    return _device_manager.to_device(data)


def to_cpu(data: Any) -> Any:
    """Move data to CPU."""
    return _device_manager.to_cpu(data)


def ensure_tensor(
    data: Union[np.ndarray, torch.Tensor],
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Ensure data is a torch tensor on the specified device."""
    if not _HAS_TORCH:
        return np.asarray(data, dtype=np.float64)

    if dtype is None:
        dtype = _device_manager.dtype
    if device is None:
        device = _device_manager.device

    if isinstance(data, torch.Tensor):
        return data.to(dtype=dtype, device=device)
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data).to(dtype=dtype, device=device)
    else:
        return torch.tensor(data, dtype=dtype, device=device)


# =============================================================================
# Caching
# =============================================================================


def mesh_hash(
    vertices: Union[np.ndarray, torch.Tensor], faces: Union[np.ndarray, torch.Tensor]
) -> str:
    """Generate unique hash for mesh vertices and faces."""
    if _HAS_TORCH and isinstance(vertices, torch.Tensor):
        v_bytes = vertices.cpu().numpy().tobytes()
        f_bytes = faces.cpu().numpy().tobytes()
    else:
        v_bytes = vertices.tobytes()
        f_bytes = faces.tobytes()
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
# Sparse Matrix Utilities (Torch-based)
# =============================================================================


def sparse_coo_matrix(
    rows: np.ndarray,
    cols: np.ndarray,
    data: np.ndarray,
    shape: Tuple[int, int],
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Create sparse COO tensor."""
    if device is None:
        device = _device_manager.device

    if not _HAS_TORCH:
        from scipy.sparse import coo_matrix

        return coo_matrix((data, (rows, cols)), shape=shape)

    indices = torch.tensor(np.stack([rows, cols]), dtype=torch.long, device=device)
    values = torch.tensor(data, dtype=_device_manager.dtype, device=device)
    return torch.sparse_coo_tensor(indices, values, shape, device=device)


def sparse_csr_matrix(
    rows: np.ndarray,
    cols: np.ndarray,
    data: np.ndarray,
    shape: Tuple[int, int],
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Create sparse CSR tensor."""
    if device is None:
        device = _device_manager.device

    if not _HAS_TORCH:
        from scipy.sparse import csr_matrix

        return csr_matrix((data, (rows, cols)), shape=shape)

    # Convert COO to CSR
    coo = sparse_coo_matrix(rows, cols, data, shape, device)
    return coo.to_sparse_csr()


def to_sparse(tensor: torch.Tensor) -> torch.Tensor:
    """Convert dense tensor to sparse if mostly zeros."""
    if tensor.dim() != 2:
        return tensor

    nnz = (tensor != 0).sum()
    density = nnz / tensor.numel()

    if density < 0.3:
        return tensor.to_sparse()
    return tensor


# =============================================================================
# Optimized Linear Algebra
# =============================================================================


def cotangent_weights(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Compute cotangent weights for all edges.

    Args:
        vertices: (n_vertices, 3) tensor
        faces: (n_faces, 3) tensor

    Returns:
        Dictionary of edge index -> cotangent weight
    """
    if not _HAS_TORCH or not isinstance(vertices, torch.Tensor):
        return _cotangent_numpy(vertices, faces)

    device = vertices.device
    n_faces = faces.shape[0]
    weights = {}

    for i in range(n_faces):
        i0, i1, i2 = faces[i].long()
        v0, v1, v2 = vertices[i0], vertices[i1], vertices[i2]

        def cot(a, b, c):
            ba = b - a
            ca = c - a
            cross = torch.linalg.cross(ba, ca)
            norm = torch.norm(cross)
            if norm < 1e-12:
                return torch.tensor(0.0, device=device)
            return torch.dot(ba, ca) / norm

        cot0 = cot(v0, v1, v2)
        cot1 = cot(v1, v2, v0)
        cot2 = cot(v2, v0, v1)

        e01 = (min(i0.item(), i1.item()), max(i0.item(), i1.item()))
        e12 = (min(i1.item(), i2.item()), max(i1.item(), i2.item()))
        e20 = (min(i2.item(), i0.item()), max(i2.item(), i0.item()))

        weights[e01] = weights.get(e01, 0.0) + cot0
        weights[e12] = weights.get(e12, 0.0) + cot1
        weights[e20] = weights.get(e20, 0.0) + cot2

    return weights


def _cotangent_numpy(vertices: np.ndarray, faces: np.ndarray) -> dict:
    """NumPy fallback for cotangent weights."""
    weights = {}
    for f in faces:
        i0, i1, i2 = f
        v0, v1, v2 = vertices[i0], vertices[i1], vertices[i2]

        def cot(a, b, c):
            ba = b - a
            ca = c - a
            cross = np.cross(ba, ca)
            norm = np.linalg.norm(cross)
            if norm < 1e-12:
                return 0.0
            return np.dot(ba, ca) / norm

        cot0 = cot(v0, v1, v2)
        cot1 = cot(v1, v2, v0)
        cot2 = cot(v2, v0, v1)

        e01 = (min(i0, i1), max(i0, i1))
        e12 = (min(i1, i2), max(i1, i2))
        e20 = (min(i2, i0), max(i2, i0))

        weights[e01] = weights.get(e01, 0.0) + cot0
        weights[e12] = weights.get(e12, 0.0) + cot1
        weights[e20] = weights.get(e20, 0.0) + cot2

    return weights


def face_areas(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Compute face areas.

    Args:
        vertices: (n_vertices, 3) tensor
        faces: (n_faces, 3) tensor

    Returns:
        (n_faces,) tensor of areas
    """
    if not _HAS_TORCH or not isinstance(vertices, torch.Tensor):
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        return 0.5 * np.linalg.norm(cross, axis=1)

    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    cross = torch.linalg.cross(v1 - v0, v2 - v0)
    return 0.5 * torch.norm(cross, dim=1)


def vertex_normals(
    vertices: torch.Tensor, faces: torch.Tensor, normalize: bool = True
) -> torch.Tensor:
    """Compute vertex normals.

    Args:
        vertices: (n_vertices, 3) tensor
        faces: (n_faces, 3) tensor
        normalize: Whether to normalize the normals

    Returns:
        (n_vertices, 3) tensor of normals
    """
    if not _HAS_TORCH or not isinstance(vertices, torch.Tensor):
        return _vertex_normals_numpy(vertices, faces, normalize)

    device = vertices.device
    n_v = vertices.shape[0]
    normals = torch.zeros(n_v, 3, dtype=vertices.dtype, device=device)

    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    cross = torch.linalg.cross(v1 - v0, v2 - v0)

    for k in range(3):
        normals.index_add_(0, faces[:, k], cross)

    if normalize:
        norms = torch.norm(normals, dim=1, keepdim=True)
        norms = torch.where(norms > 1e-14, norms, torch.ones_like(norms))
        normals = normals / norms

    return normals


def _vertex_normals_numpy(
    vertices: np.ndarray, faces: np.ndarray, normalize: bool = True
) -> np.ndarray:
    """NumPy fallback for vertex normals."""
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
    vertices: torch.Tensor, faces: torch.Tensor, method: str = "barycentric"
) -> torch.Tensor:
    """Compute vertex areas (barycentric or voronoi).

    Args:
        vertices: (n_vertices, 3) tensor
        faces: (n_faces, 3) tensor
        method: 'barycentric' or 'voronoi'

    Returns:
        (n_vertices,) tensor of areas
    """
    if not _HAS_TORCH or not isinstance(vertices, torch.Tensor):
        return _vertex_areas_numpy(vertices, faces, method)

    device = vertices.device
    n_v = vertices.shape[0]
    n_f = faces.shape[0]
    areas = torch.zeros(n_v, dtype=vertices.dtype, device=device)

    face_areas_val = face_areas(vertices, faces)

    if method == "barycentric":
        for k in range(3):
            areas.index_add_(0, faces[:, k], face_areas_val / 3.0)
    else:
        # Mixed Voronoi (simplified)
        for k in range(3):
            areas.index_add_(0, faces[:, k], face_areas_val / 2.0)

    return torch.clamp(areas, min=1e-12)


def _vertex_areas_numpy(
    vertices: np.ndarray, faces: np.ndarray, method: str = "barycentric"
) -> np.ndarray:
    """NumPy fallback for vertex areas."""
    n_v = vertices.shape[0]
    areas = np.zeros(n_v)
    face_areas_val = face_areas(vertices, faces)

    if method == "barycentric":
        for k in range(3):
            np.add.at(areas, faces[:, k], face_areas_val / 3.0)
    else:
        for k in range(3):
            np.add.at(areas, faces[:, k], face_areas_val / 2.0)

    return np.maximum(areas, 1e-12)


# =============================================================================
# Parallel Processing
# =============================================================================


def parallel_map(
    func: Callable, items: list, n_jobs: int = -1, device: Optional[str] = None
) -> list:
    """Parallel map with optional GPU distribution."""
    if n_jobs == -1:
        n_jobs = mp.cpu_count()

    if n_jobs == 1:
        return [func(item) for item in items]

    if device == "cuda" and _HAS_TORCH and torch.cuda.device_count() > 1:
        return _parallel_cuda(func, items)

    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        return list(executor.map(func, items))


def _parallel_cuda(func: Callable, items: list) -> list:
    """Distribute work across multiple CUDA devices."""
    n_devices = torch.cuda.device_count()
    chunks = np.array_split(items, n_devices)

    results = []
    for i, chunk in enumerate(chunks):
        with torch.cuda.device(i):
            results.extend([func(item) for item in chunk])

    return results


# =============================================================================
# Memory Management
# =============================================================================


class MemoryPool:
    """Pre-allocated tensor pool for reduce allocations."""

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or _device_manager.device
        self._pools = {}

    def get(self, shape: Tuple[int, ...], dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Get a tensor from the pool or create new one."""
        if dtype is None:
            dtype = _device_manager.dtype

        key = (shape, dtype)

        if key in self._pools and self._pools[key]:
            return self._pools[key].pop()

        return torch.zeros(shape, dtype=dtype, device=self.device)

    def release(self, tensor: torch.Tensor):
        """Return tensor to pool."""
        key = (tuple(tensor.shape), tensor.dtype)
        if key not in self._pools:
            self._pools[key] = []

        tensor.fill_(0)
        self._pools[key].append(tensor)

    def clear(self):
        """Clear all pools."""
        self._pools.clear()


_memory_pool = MemoryPool()


def get_tensor_pool() -> MemoryPool:
    """Get the global tensor pool."""
    return _memory_pool


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
        if _HAS_TORCH:
            torch.cuda.synchronize() if torch.cuda.is_available() else None
        import time

        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        if _HAS_TORCH:
            torch.cuda.synchronize() if torch.cuda.is_available() else None
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

    if _HAS_TORCH:
        torch.cuda.synchronize() if torch.cuda.is_available() else None

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = func(*args, **kwargs)

        if _HAS_TORCH:
            torch.cuda.synchronize() if torch.cuda.is_available() else None

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
        "has_torch": _HAS_TORCH,
        "torch_version": _TORCH_VERSION if _HAS_TORCH else None,
        "n_cpus": mp.cpu_count(),
        "has_cuda": _device_manager.has_cuda,
        "has_mps": _device_manager.has_mps,
    }

    if _HAS_TORCH and torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["n_gpus"] = torch.cuda.device_count()
        for i in range(torch.cuda.device_count()):
            info[f"gpu_{i}"] = torch.cuda.get_device_name(i)

    if _HAS_PSUTIL:
        mem = psutil.virtual_memory()
        info["ram_gb"] = mem.total / (1024**3)
        info["available_ram_gb"] = mem.available / (1024**3)

    return info


# =============================================================================
# Gradient utilities for optimization
# =============================================================================


def gradient_descent(mesh, energy_fn, lr: float = 0.01, steps: int = 100) -> torch.Tensor:
    """Simple gradient descent on mesh vertices.

    Args:
        mesh: Mesh object with vertices tensor
        energy_fn: Function that computes energy from vertices
        lr: Learning rate
        steps: Number of iterations

    Returns:
        Optimized vertices
    """
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch required for gradient descent")

    vertices = mesh.vertices.clone()
    vertices.requires_grad_(True)

    optimizer = torch.optim.SGD([vertices], lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()
        energy = energy_fn(vertices)
        energy.backward()
        optimizer.step()

    return vertices.detach()


def compute_gradients(output: torch.Tensor, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
    """Compute gradients of output with respect to inputs."""
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch required for gradient computation")

    grads = torch.autograd.grad(output, inputs, create_graph=True)
    return grads


__all__ = [
    "DeviceManager",
    "get_device",
    "has_cuda",
    "has_mps",
    "set_device",
    "to_device",
    "to_cpu",
    "ensure_tensor",
    "mesh_hash",
    "lru_cache",
    "cached_property",
    "sparse_coo_matrix",
    "sparse_csr_matrix",
    "to_sparse",
    "cotangent_weights",
    "face_areas",
    "vertex_normals",
    "vertex_areas",
    "parallel_map",
    "MemoryPool",
    "get_tensor_pool",
    "Timer",
    "benchmark",
    "get_system_info",
    "gradient_descent",
    "compute_gradients",
]
