"""
Example 11 — PDE: Hodge Decomposition, Fluids, and Cloth
==========================================================
Decomposing vector fields, incompressible flow from vorticity,
and mass-spring cloth simulation.

Modules used:
    ddglearn.pde.hodge_decomposition — hodge_decomposition,
                                        is_divergence_free,
                                        is_curl_free
    ddglearn.pde.fluids              — fluid_velocity_from_vorticity
    ddglearn.pde.cloth               — cloth_simulation_step
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from ddglearn.core.halfedge import HalfEdgeMesh
from ddglearn.core.mesh_io import load_mesh
from ddglearn.pde.hodge_decomposition import (
    hodge_decomposition,
    is_divergence_free,
    is_curl_free,
)
from ddglearn.pde.fluids import fluid_velocity_from_vorticity
from ddglearn.pde.cloth import cloth_simulation_step

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
vertices, faces = load_mesh(os.path.join(DATA, "teapot.obj"))
vertices = vertices / (vertices.max(axis=0) - vertices.min(axis=0)).max()
mesh = HalfEdgeMesh(vertices, faces)

# ---------------------------------------------------------------------------
# 1. Hodge decomposition: v = grad(f) + *dω + harmonic
# ---------------------------------------------------------------------------
# Build a random edge vector field
np.random.seed(0)
v_edges = np.random.randn(mesh.n_edges)

grad_f, div_free, harmonic = hodge_decomposition(mesh, v_edges)
print("Hodge decomposition:")
print(f"  grad_f:    shape={grad_f.shape}, max={np.abs(grad_f).max():.4f}")
print(f"  div_free:  shape={div_free.shape}, max={np.abs(div_free).max():.4f}")
print(f"  harmonic:  shape={harmonic.shape}, max={np.abs(harmonic).max():.4f}")

# Reconstruct: v = grad_f + div_free + harmonic
v_recon = grad_f + div_free + harmonic
print(f"  Reconstruction error: {np.abs(v_edges - v_recon).max():.2e}")

# ---------------------------------------------------------------------------
# 2. Divergence-free / curl-free checks
# ---------------------------------------------------------------------------
print(f"\nIs grad_f divergence-free? {is_divergence_free(mesh, grad_f)}")
print(f"Is grad_f curl-free?      {is_curl_free(mesh, grad_f)}")
print(f"Is div_free curl-free?    {is_curl_free(mesh, div_free)}")

# ---------------------------------------------------------------------------
# 3. Fluid velocity from vorticity
# ---------------------------------------------------------------------------
omega = np.random.RandomState(42).randn(mesh.n_vertices)
v_flux, omega_diff = fluid_velocity_from_vorticity(mesh, omega, viscosity=0.01, dt=0.1)
print(f"\nFluid from vorticity:")
print(f"  Velocity flux: shape={v_flux.shape}, max={np.abs(v_flux).max():.4f}")
print(f"  Diffused ω:    shape={omega_diff.shape}, max={np.abs(omega_diff).max():.4f}")

# ---------------------------------------------------------------------------
# 4. Cloth simulation (mass-spring step)
# ---------------------------------------------------------------------------
vertices_c, faces_c = load_mesh(os.path.join(DATA, "humanoid_tri.obj"))
vertices_c = vertices_c / (vertices_c.max(axis=0) - vertices_c.min(axis=0)).max()
cloth = HalfEdgeMesh(vertices_c, faces_c)

n = cloth.n_vertices
pos = cloth.vertices.copy()
vel = np.zeros_like(pos)
dt = 0.01

# Pin top vertices (z > max_z - 10%)
z_max = pos[:, 2].max()
pinned = np.where(pos[:, 2] > z_max - 0.1 * (z_max - pos[:, 2].min()))[0]
print(f"\nCloth simulation: {cloth.n_vertices}V, pinning {len(pinned)} top vertices")

pos_new, vel_new = cloth_simulation_step(
    cloth, pos, vel, dt=dt, mass=1.0, stiffness=1000.0,
    damping=0.99, gravity=-9.81, pinned_vertices=pinned.tolist()
)

disp = np.linalg.norm(pos_new - pos, axis=1)
print(f"  Max displacement: {disp.max():.6f}")
print(f"  Vel max: {np.abs(vel_new).max():.6f}")

print("\nDone.")
