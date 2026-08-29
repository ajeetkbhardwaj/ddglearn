# Tutorial: Simulating Fluids on Surfaces

**Problem:** How does water flow over a curved surface? Standard 2D fluid sim doesn't account for surface curvature. We need to solve the Navier-Stokes equations directly on the manifold.

**Solution:** The vorticity formulation of incompressible Euler equations on surfaces, solved via DEC.

## Step 1: Set up the mesh

```python
from ddglearn.core.mesh_io import load_mesh
from ddglearn.core.halfedge import HalfEdgeMesh
import numpy as np

vertices, faces = load_mesh("data/bunny.obj")
mesh = HalfEdgeMesh(vertices, faces)
```

## Step 2: Define initial vorticity

Place a vortex at a vertex on the bunny's back:

```python
omega = np.zeros(mesh.n_vertices)
# Gaussian bump of vorticity at vertex 1000
center = 1000
for i in range(mesh.n_vertices):
    diff = vertices[i] - vertices[center]
    omega[i] = np.exp(-np.dot(diff, diff) / 0.005)
```

Visualize the initial vorticity:

```python
from ddglearn.core.visualization import save_mesh_snapshot

save_mesh_snapshot(vertices, faces, omega,
    filename="tutorial_fluid_vorticity.html",
    title="Initial vorticity field")
```

## Step 3: Compute velocity from vorticity

The key insight: in 2D/incompressible flow, velocity is determined by vorticity via a Poisson solve for the stream function, then taking the gradient:

```python
from ddglearn.pde.fluids import fluid_velocity_from_vorticity

flux, omega_diffused = fluid_velocity_from_vorticity(mesh, omega, viscosity=0.0, dt=0.01)
print(f"Velocity flux shape: {flux.shape}  (one value per edge)")
print(f"Max velocity: {np.abs(flux).max():.4f}")
```

## Step 4: Animate the flow

Advect the vorticity field to see the fluid evolve:

```python
from ddglearn.pde.fluids import fluid_velocity_from_vorticity
from ddglearn.core.visualization import save_mesh_animation

frames = []
omega_frame = omega.copy()

for step in range(100):
    frames.append(omega_frame.copy())

    # Compute velocity from current vorticity
    flux, omega_frame = fluid_velocity_from_vorticity(
        mesh, omega_frame, viscosity=0.001, dt=0.01
    )

frames = np.array(frames)
save_mesh_animation(
    vertices, faces, frames,
    filename="tutorial_fluid_animation.html",
    title="Surface fluid simulation",
    colormap="coolwarm",
    fps=15,
)
```

```bash
open tutorial_fluid_animation.html
```

## Step 5: Add viscosity

Viscosity causes the vorticity to diffuse over time — the flow gradually slows down:

```python
frames_visc = []
omega_visc = omega.copy()

for step in range(100):
    frames_visc.append(omega_visc.copy())
    flux, omega_visc = fluid_velocity_from_vorticity(
        mesh, omega_visc, viscosity=0.01, dt=0.01  # 10x more viscosity
    )

save_mesh_animation(
    vertices, faces, np.array(frames_visc),
    filename="tutorial_fluid_viscous.html",
    title="Viscous surface fluid",
    fps=15,
)
```

## How it works

The surface fluid solver uses three DEC operators:

1. **Poisson solve:** $\Delta \psi = \omega$ — find stream function from vorticity
2. **Gradient:** $d\psi$ — compute 1-form from stream function
3. **Hodge star \*1:** Rotate gradient by 90° in the tangent plane — get velocity flux

This is the discrete equivalent of $V = \nabla^\perp \psi$.

## Complete script

```python
"""Surface fluid simulation tutorial — Euler flow on a bunny."""
import numpy as np
from ddglearn.core.mesh_io import load_mesh
from ddglearn.core.halfedge import HalfEdgeMesh
from ddglearn.pde.fluids import fluid_velocity_from_vorticity
from ddglearn.core.visualization import save_mesh_animation, save_mesh_snapshot

vertices, faces = load_mesh("data/bunny.obj")
mesh = HalfEdgeMesh(vertices, faces)

# Initial vorticity: Gaussian at vertex 1000
omega = np.zeros(mesh.n_vertices)
center = 1000
for i in range(mesh.n_vertices):
    diff = vertices[i] - vertices[center]
    omega[i] = np.exp(-np.dot(diff, diff) / 0.005)

save_mesh_snapshot(vertices, faces, omega,
    filename="tutorial_fluid_vorticity.html", title="Initial vorticity")

# Animate
frames = []
w = omega.copy()
for _ in range(100):
    frames.append(w.copy())
    flux, w = fluid_velocity_from_vorticity(mesh, w, viscosity=0.001, dt=0.01)

save_mesh_animation(vertices, faces, np.array(frames),
    filename="tutorial_fluid_animation.html",
    title="Surface fluid", fps=15)
```
