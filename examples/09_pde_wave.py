"""
Example 09 -- PDE: Wave Equation on a Surface
=============================================
Simulating wave propagation on a triangulated mesh using an
implicit Crank-Nicolson scheme (unconditionally stable).

Modules used:
    ddglearn.pde.wave        -- wave_step
    ddglearn.core.visualization -- save_mesh_animation
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from ddglearn.core.halfedge import HalfEdgeMesh
from ddglearn.core.mesh_io import load_mesh
from ddglearn.pde.wave import wave_step
from ddglearn.operators.hodge_star import hodge_star_0, hodge_star_1
from ddglearn.operators.exterior_derivative import d0

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
OUT = os.path.join(os.path.dirname(__file__), "_output")
os.makedirs(OUT, exist_ok=True)

vertices, faces = load_mesh(os.path.join(DATA, "bunny.obj"))
vertices = vertices / (vertices.max(axis=0) - vertices.min(axis=0)).max()
mesh = HalfEdgeMesh(vertices, faces)

n = mesh.n_vertices
H0 = hodge_star_0(mesh)
H1 = hodge_star_1(mesh)
D0 = d0(mesh)
W = D0.T @ H1 @ D0


def wave_energy(u_prev, u_curr, dt):
    v = (u_curr - u_prev) / dt
    ke = 0.5 * float(v @ (H0 @ v))
    pe = 0.5 * float(u_curr @ (W @ u_curr))
    return ke + pe


# ---------------------------------------------------------------------------
# 1. Single wave step
# ---------------------------------------------------------------------------
u_prev = np.zeros(n)
center = vertices[n // 2]
dists = np.linalg.norm(vertices - center, axis=1)
u_curr = np.exp(-dists**2 / (2 * 0.05**2))

dt = 0.005
u_next = wave_step(mesh, u_prev, u_curr, dt, pin_index=0)
print(f"Single wave step:")
print(f"  u_prev max: {np.abs(u_prev).max():.6f}")
print(f"  u_curr max: {np.abs(u_curr).max():.6f}")
print(f"  u_next max: {np.abs(u_next).max():.6f}")

# ---------------------------------------------------------------------------
# 2. Multi-step wave simulation
# ---------------------------------------------------------------------------
n_steps = 200
u_prev = np.zeros(n)
u_curr = u_next.copy()
history = [u_prev.copy(), u_curr.copy()]

E0 = wave_energy(u_prev, u_curr, dt)
print(f"\nWave simulation ({n_steps} steps, dt={dt}):")
print(f"  Initial energy: {E0:.4f}")

for step in range(n_steps):
    u_next = wave_step(mesh, u_prev, u_curr, dt, pin_index=0)
    u_prev, u_curr = u_curr, u_next
    history.append(u_curr.copy())

history = np.array(history)
E_end = wave_energy(history[-2], history[-1], dt)
print(f"  Final energy:  {E_end:.4f}")
print(f"  Energy ratio:  {E_end / E0:.6f}")
print(f"\n  State at key frames:")
for i in [0, 1, n_steps // 4, n_steps // 2, 3 * n_steps // 4, len(history) - 1]:
    E_i = wave_energy(history[max(i - 1, 0)], history[i], dt) if i > 0 else E0
    print(f"    step {i:3d}: max|u|={np.abs(history[i]).max():.6f}  E={E_i:.4f}")

# ---------------------------------------------------------------------------
# 3. Save animation
# ---------------------------------------------------------------------------
from ddglearn.core.visualization import save_mesh_animation

print("\nSaving animation...")
save_mesh_animation(vertices, faces, history,
                    os.path.join(OUT, "09_wave.html"),
                    title="Wave Equation (Crank-Nicolson)",
                    colormap="coolwarm", fps=15)

print("\nDone.")
