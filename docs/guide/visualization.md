# Visualization

Standalone HTML visualization with WebGL rendering and interactive controls.

## Saving Animations

```python
from ddglearn.core.visualization import save_mesh_animation

save_mesh_animation(
    vertices, faces,
    frames,                    # (n_frames, n_verts) vertex scalar values
    filename="animation.html",
    title="Heat Diffusion",
    colormap="coolwarm",
    vmin=0.0, vmax=1.0,       # optional color range
    fps=10,
)
```

Opens a standalone HTML file in the browser with:

- WebGL-rendered mesh with per-vertex coloring
- Time slider for scrubbing frames
- Play/pause button

## Saving Snapshots

```python
from ddglearn.core.visualization import save_mesh_snapshot

save_mesh_snapshot(
    vertices, faces,
    colors,                    # (n_verts,) scalar values or (n_verts, 3) RGB
    filename="snapshot.html",
    title="Geodesic Distance",
    colormap="coolwarm",
)
```

## Example: Heat Diffusion Animation

```python
import numpy as np
from ddglearn.core.mesh_io import load_mesh
from ddglearn.core.halfedge import HalfEdgeMesh
from ddglearn.core.visualization import save_mesh_animation
from ddglearn.pde.heat import implicit_heat

vertices, faces = load_mesh("data/bunny.obj")
mesh = HalfEdgeMesh(vertices, faces)

u0 = np.zeros(mesh.n_vertices)
u0[0] = 1.0  # heat source at vertex 0

# Collect frames
frames = []
u = u0.copy()
for i in range(50):
    frames.append(u.copy())
    u = implicit_heat(mesh, u, t=1e-5, steps=1)

frames = np.array(frames)
save_mesh_animation(vertices, faces, frames, filename="heat.html")
```

## In Jupyter

`meshplot` can also be used directly in notebooks:

```python
import meshplot as mp
p = mp.plot(vertices, faces, c=colors)
p.add_histogram(colors)
```
