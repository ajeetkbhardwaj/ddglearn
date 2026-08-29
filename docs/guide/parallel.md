# Parallel Processing

CPU parallelism for batched independent computations.

## Quick Usage

```python
from ddglearn.parallel.pool import parallel_map, thread_map, process_map

# Thread-based (good for numpy/BLAS-bound or I/O-bound work)
results = parallel_map(my_func, items, n_jobs=-1)

# Process-based (good for CPU-bound, picklable functions)
results = process_map(my_func, items, n_jobs=-1)

# Fire-and-forget (side effects only)
parallel_for(save_file, items, n_jobs=-1)
```

## Reusable Pool

```python
from ddglearn.parallel.pool import ParallelExecutor

with ParallelExecutor(n_jobs=4) as executor:
    results = executor.map(process_item, items)
    futures = [executor.submit(heavy_func, arg) for arg in args]
    results = executor.gather(futures)
```

## When to Parallelize

| Scenario | Use parallelism? |
|---|---|
| Batch of independent heavy computations (e.g., geodesics from 100 sources) | Yes — `process_map` |
| Single large matrix multiply | No — numpy BLAS already uses all cores |
| I/O-bound work (file reads, network) | Yes — `thread_map` |
| Sequential time-stepping (each step depends on previous) | No |
| Small arrays / few items | No — overhead exceeds gain |

## System Info

```python
from ddglearn.parallel.pool import get_cpu_count, available_memory_gb

n_cpus = get_cpu_count()
ram_gb = available_memory_gb()  # requires psutil
```

## Key Constraint

`process_map` and `ProcessPoolExecutor` require **picklable** callables — top-level functions, no lambdas or closures.
