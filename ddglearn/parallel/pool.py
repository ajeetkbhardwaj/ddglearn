"""CPU parallel processing utilities for heavy DDG workloads.

This module provides thread- and process-based parallelism for the
CPU-only (numpy/scipy) backend. Use :func:`parallel_map` /
:func:`process_map` for embarrassingly parallel jobs (per-mesh
reductions, batched solves, Monte-Carlo sampling, batch I/O) and
:class:`ParallelExecutor` when you want a reusable pool.

Note: :func:`process_map` and the process-based executor dispatch work to
child processes via ``multiprocessing``, so the callable and its
arguments must be picklable (define ``func`` at module level and avoid
closures/lambdas when using process-based parallelism).
"""

import os
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from typing import Callable, Iterable, List, Optional

try:
    import psutil

    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False
    psutil = None


def get_cpu_count() -> int:
    """Return the number of logical CPUs available."""
    return os.cpu_count() or 1


def available_memory_gb() -> Optional[float]:
    """Return available system RAM in GB, or ``None`` if psutil is missing."""
    if not _HAS_PSUTIL:
        return None
    return psutil.virtual_memory().available / (1024**3)


def _resolve_jobs(n_jobs: int) -> int:
    if n_jobs == -1:
        n_jobs = get_cpu_count()
    return max(1, min(n_jobs, get_cpu_count()))


def parallel_map(
    func: Callable,
    items: Iterable,
    n_jobs: int = -1,
    use_processes: bool = False,
) -> list:
    """Map ``func`` over ``items`` using a thread or process pool.

    Args:
        func: Callable applied to each item.
        items: Iterable of inputs.
        n_jobs: Number of workers (``-1`` uses all CPUs). Falls back to a
            serial loop when ``n_jobs == 1``.
        use_processes: If ``True`` use ``multiprocessing`` for true CPU
            parallelism on heavy Python/numpy loops; otherwise use threads
            (useful when the workload releases the GIL, e.g. BLAS calls).

    Returns:
        List of results in the same order as ``items``.
    """
    items = list(items)
    n_jobs = _resolve_jobs(n_jobs)

    if n_jobs == 1:
        return [func(item) for item in items]

    executor_cls = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    with executor_cls(max_workers=n_jobs) as executor:
        return list(executor.map(func, items))


def thread_map(func: Callable, items: Iterable, n_jobs: int = -1) -> list:
    """Convenience wrapper around :func:`parallel_map` using threads."""
    return parallel_map(func, items, n_jobs=n_jobs, use_processes=False)


def process_map(
    func: Callable,
    items: Iterable,
    n_jobs: int = -1,
    chunksize: int = 1,
) -> list:
    """Map ``func`` over ``items`` using a process pool (true CPU parallelism).

    Args:
        func: Top-level callable (must be picklable).
        items: Iterable of inputs.
        n_jobs: Number of worker processes (``-1`` uses all CPUs).
        chunksize: Items per worker task (tune for many small items).

    Returns:
        List of results in the same order as ``items``.
    """
    items = list(items)
    n_jobs = _resolve_jobs(n_jobs)

    if n_jobs == 1:
        return [func(item) for item in items]

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        return list(executor.map(func, items, chunksize=chunksize))


def parallel_for(
    func: Callable,
    items: Iterable,
    n_jobs: int = -1,
    use_processes: bool = False,
) -> None:
    """Fire-and-forget parallel map whose return values are discarded.

    Useful for side-effecting work such as writing files or accumulating
    results into an external store.
    """
    items = list(items)
    n_jobs = _resolve_jobs(n_jobs)

    if n_jobs == 1:
        for item in items:
            func(item)
        return

    executor_cls = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    with executor_cls(max_workers=n_jobs) as executor:
        list(executor.map(func, items))


class ParallelExecutor:
    """Reusable pool context manager.

    Example:
        with ParallelExecutor(n_jobs=4, use_processes=True) as pool:
            results = pool.map(heavy_func, meshes)
            futures = [pool.submit(heavy_func, m) for m in meshes]
    """

    def __init__(self, n_jobs: int = -1, use_processes: bool = False):
        self.n_jobs = n_jobs
        self.use_processes = use_processes
        self._executor = None

    def __enter__(self) -> "ParallelExecutor":
        n_jobs = _resolve_jobs(self.n_jobs)
        executor_cls = (
            ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
        )
        self._executor = executor_cls(max_workers=n_jobs)
        return self

    def map(self, func: Callable, items: Iterable) -> list:
        return list(self._executor.map(func, items))

    def submit(self, func: Callable, *args, **kwargs):
        return self._executor.submit(func, *args, **kwargs)

    def gather(self, futures) -> list:
        return [f.result() for f in as_completed(futures)]

    def __exit__(self, *args) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None


__all__ = [
    "get_cpu_count",
    "available_memory_gb",
    "parallel_map",
    "thread_map",
    "process_map",
    "parallel_for",
    "ParallelExecutor",
]
