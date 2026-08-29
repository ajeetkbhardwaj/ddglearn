"""
Example 08 -- PDE: Poisson Equation and Heat Diffusion
======================================================
Solving PDEs on surfaces using DEC.

Modules used:
    ddglearn.pde.poisson -- solve_poisson
    ddglearn.pde.heat    -- implicit_heat_step, implicit_heat
    ddglearn.core.visualization -- save_mesh_animation
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from ddglearn.core.halfedge import HalfEdgeMesh
from ddglearn.core.mesh_io import load_mesh
from ddglearn.pde.poisson import solve_poisson
from ddglearn.pde.heat import implicit_heat_step

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
OUT = os.path.join(os.path.dirname(__file__), "_output")
os.makedirs(OUT, exist_ok=True)

vertices, faces = load_mesh(os.path.join(DATA, "bunny.obj"))
vertices = vertices / (vertices.max(axis=0) - vertices.min(axis=0)).max()
mesh = HalfEdgeMesh(vertices, faces)

# ---------------------------------------------------------------------------
# 1. Poisson equation:  Delta u = f   with Dirichlet BC
# ---------------------------------------------------------------------------
center = vertices[0]
dists = np.linalg.norm(vertices - center, axis=1)
f = np.exp(-dists**2 / (2 * 0.01**2))
f /= f.sum()

u_poisson = solve_poisson(mesh, f, pin_index=0, pin_value=0.0)
print(f"Poisson solution: shape={u_poisson.shape}")
print(f"  min={u_poisson.min():.6f}, max={u_poisson.max():.6f}")
print(f"  |u(pin)| = {abs(u_poisson[0]):.2e}")

# ---------------------------------------------------------------------------
# 2. Multi-step heat diffusion
# ---------------------------------------------------------------------------
u0 = f.copy()
t = 1e-4
n_steps = 500
heat_history = [u0.copy()]
u = u0.copy()
for _ in range(n_steps):
    u = implicit_heat_step(mesh, u, t, pin_index=0)
    heat_history.append(u.copy())
heat_history = np.array(heat_history)

print(f"\nHeat diffusion ({n_steps} steps, t={t}):")
print(f"  min={u.min():.6f}, max={u.max():.6f}")
print(f"  Initial max: {u0.max():.6f}")
print(f"  Final max:   {u.max():.6f}")
print(f"  Heat spread: {u.max() < u0.max()}")

# ---------------------------------------------------------------------------
# 3. Poisson with Neumann BC
# ---------------------------------------------------------------------------
f2 = np.random.RandomState(42).randn(mesh.n_vertices)
f2 -= f2.mean()
u_neumann = solve_poisson(mesh, f2, pin_index=0)
print(f"\nPoisson (natural BC): min={u_neumann.min():.6f}, max={u_neumann.max():.6f}")

# ---------------------------------------------------------------------------
# 4. Save animations / snapshots
# ---------------------------------------------------------------------------
from ddglearn.core.visualization import save_mesh_animation, save_mesh_snapshot

print("\nSaving visualizations...")
save_mesh_snapshot(vertices, faces, u_poisson,
                   os.path.join(OUT, "08_poisson.html"), title="Poisson solution")
save_mesh_snapshot(vertices, faces, u_neumann,
                   os.path.join(OUT, "08_poisson_neumann.html"), title="Poisson (Neumann BC)")
save_mesh_animation(vertices, faces, heat_history,
                    os.path.join(OUT, "08_heat_diffusion.html"),
                    title="Heat Diffusion", colormap="hot", fps=5)

print("\nDone.")
