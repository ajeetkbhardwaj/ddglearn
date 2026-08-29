# Tutorial: Measuring Distances on Curved Surfaces

**Problem:** Euclidean distance is wrong on curved surfaces — the shortest path between two points on a bunny's ear follows the surface, not a straight line through the ear. We need *geodesic distance*.

**Solution:** The Heat Method (Crane et al. 2013) computes geodesic distances efficiently by solving three simple PDEs.

## Step 1: Load the mesh

```python
from ddglearn.core.mesh_io import load_mesh
from ddglearn.core.halfedge import HalfEdgeMesh

vertices, faces = load_mesh("data/bunny.obj")
mesh = HalfEdgeMesh(vertices, faces)
print(f"Loaded bunny: {mesh.n_vertices} vertices, {mesh.n_faces} faces")
```

## Step 2: Compute geodesic from one vertex

Pick a vertex on the bunny's nose as the source:

```python
from ddglearn.pde.geodesics import geodesic_distance

source = 500  # nose vertex
d = geodesic_distance(mesh, source_indices=[source], time_scale=1e-3)
print(f"Max geodesic distance: {d.max():.4f}")
```

Visualize the distance field — red is close to the source, blue is far:

```python
from ddglearn.core.visualization import save_mesh_snapshot

save_mesh_snapshot(
    vertices, faces, d,
    filename="tutorial_geodesic_single.html",
    title=f"Geodesic from vertex {source}",
    colormap="coolwarm",
)
```

```bash
open tutorial_geodesic_single.html
```

## Step 3: Multi-source geodesics

Find the distance from multiple landmarks simultaneously — which vertex is closest to each landmark?

```python
import numpy as np
from ddglearn.core.visualization import save_mesh_snapshot

# Pick 3 landmarks: nose, left ear tip, right ear tip
landmarks = [500, 1200, 800]

distances = []
for src in landmarks:
    d = geodesic_distance(mesh, source_indices=[src], time_scale=1e-3)
    distances.append(d)

# For each vertex, find the nearest landmark
distances = np.array(distances)  # (3, n_verts)
nearest_landmark = np.argmin(distances, axis=0).astype(float)

save_mesh_snapshot(
    vertices, faces, nearest_landmark,
    filename="tutorial_geodesic_multisource.html",
    title="Nearest landmark (0=nose, 1=left ear, 2=right ear)",
    colormap="Set1",
)
```

## Step 4: Compare methods

The Heat Method supports three approaches — `poisson` (standard), `varadhan` (log-formula), and `fmm_graph` (Dijkstra):

```python
import time

methods = ["poisson", "varadhan", "fmm_graph"]
for method in methods:
    t0 = time.time()
    d = geodesic_distance(mesh, source_indices=[source], time_scale=1e-3, method=method)
    dt = time.time() - t0
    print(f"  {method:12s}  max={d.max():.4f}  time={dt*1000:.1f}ms")
```

**When to use which:**

| Method | Speed | Accuracy | Best for |
|---|---|---|---|
| `poisson` | Fast | Good | General purpose |
| `varadhan` | Fast | Good | Closed surfaces (no boundary) |
| `fmm_graph` | Fastest | Exact graph distance | When you want Euclidean edge lengths |

## Step 5: Geodesic as a tool for analysis

Geodesic distance reveals surface structure that Euclidean distance misses:

```python
import numpy as np

# Two vertices that are close in Euclidean space but far on the surface
v1, v2 = 500, 1500
euclidean = np.linalg.norm(vertices[v1] - vertices[v2])
geodesic = geodesic_distance(mesh, source_indices=[v1], time_scale=1e-3)[v2]
print(f"Euclidean: {euclidean:.4f}")
print(f"Geodesic:  {geodesic:.4f}")
print(f"Ratio:     {geodesic/euclidean:.1f}x")
```

If the ratio is large, the two points are close in 3D space but the path must go around a feature (like an ear or limb).

## Complete script

```python
"""Geodesic distance tutorial — measuring distances on curved surfaces."""
import numpy as np
from ddglearn.core.mesh_io import load_mesh
from ddglearn.core.halfedge import HalfEdgeMesh
from ddglearn.pde.geodesics import geodesic_distance
from ddglearn.core.visualization import save_mesh_snapshot

vertices, faces = load_mesh("data/bunny.obj")
mesh = HalfEdgeMesh(vertices, faces)

# Single source
source = 500
d = geodesic_distance(mesh, source_indices=[source], time_scale=1e-3)
save_mesh_snapshot(vertices, faces, d,
    filename="tutorial_geodesic_single.html",
    title=f"Geodesic from vertex {source}")

# Multi-source
landmarks = [500, 1200, 800]
distances = np.array([
    geodesic_distance(mesh, [s], time_scale=1e-3) for s in landmarks
])
nearest = np.argmin(distances, axis=0).astype(float)
save_mesh_snapshot(vertices, faces, nearest,
    filename="tutorial_geodesic_multisource.html",
    title="Nearest landmark", colormap="Set1")

# Compare Euclidean vs geodesic
euclidean = np.linalg.norm(vertices[500] - vertices[1500])
geodesic = geodesic_distance(mesh, [500], time_scale=1e-3)[1500]
print(f"Euclidean: {euclidean:.4f}  Geodesic: {geodesic:.4f}  Ratio: {geodesic/euclidean:.1f}x")
```
