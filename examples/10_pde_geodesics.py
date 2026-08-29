"""
Example 10 -- PDE: Geodesic Distance via the Heat Method
=========================================================
Computing shortest-path distances on surfaces using
heat diffusion (Crane, Weischedel, Wardetzky 2013).

Modules used:
    ddglearn.pde.geodesics       -- solve_heat_diffusion,
                                     geodesic_distance,
                                     heat_method_geodesics
    ddglearn.core.visualization  -- save_mesh_snapshot
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from ddglearn.core.halfedge import HalfEdgeMesh
from ddglearn.core.mesh_io import load_mesh
from ddglearn.pde.geodesics import (
    solve_heat_diffusion,
    geodesic_distance,
)
from ddglearn.core.visualization import save_mesh_snapshot

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
OUT = os.path.join(os.path.dirname(__file__), "_output")
os.makedirs(OUT, exist_ok=True)

vertices, faces = load_mesh(os.path.join(DATA, "teapot.obj"))
vertices = vertices / (vertices.max(axis=0) - vertices.min(axis=0)).max()
mesh = HalfEdgeMesh(vertices, faces)

ts = 1e-2

# ---------------------------------------------------------------------------
# 1. Heat diffusion from a single source
# ---------------------------------------------------------------------------
source = [0]
u_heat = solve_heat_diffusion(mesh, source, time_scale=ts)
print(f"Heat diffusion from vertex {source[0]}:")
print(f"  shape={u_heat.shape}, min={u_heat.min():.6f}, max={u_heat.max():.6f}")

# ---------------------------------------------------------------------------
# 2. Full geodesic distance (heat method)
# ---------------------------------------------------------------------------
dist_poisson = geodesic_distance(mesh, source, time_scale=ts, method="poisson")
print(f"\nGeodesic distance (Poisson method):")
print(f"  shape={dist_poisson.shape}")
print(f"  min={dist_poisson.min():.6f}, max={dist_poisson.max():.6f}")
print(f"  dist[source] = {dist_poisson[source[0]]:.6f} (should be ~0)")

# ---------------------------------------------------------------------------
# 3. Varadhan's formula
# ---------------------------------------------------------------------------
dist_var = geodesic_distance(mesh, source, time_scale=ts, method="varadhan")
print(f"\nGeodesic distance (Varadhan):")
print(f"  max diff vs Poisson: {np.abs(dist_poisson - dist_var).max():.4f}")

# ---------------------------------------------------------------------------
# 4. Multi-source geodesics (batch mode)
# ---------------------------------------------------------------------------
sources = [0, mesh.n_vertices // 2, mesh.n_vertices - 1]
geo = geodesic_distance(mesh, sources, time_scale=ts, method="poisson")
print(f"\nMulti-source geodesics:")
print(f"  shape={geo.shape}")
print(f"  min={geo.min():.6f}, max={geo.max():.6f}")

# ---------------------------------------------------------------------------
# 5. Save snapshots
# ---------------------------------------------------------------------------
print("\nSaving visualizations...")
save_mesh_snapshot(vertices, faces, u_heat,
                   os.path.join(OUT, "10_heat_diffusion.html"),
                   title="Heat diffusion from vertex 0", colormap="hot")

for idx, s in enumerate(sources):
    d = geodesic_distance(mesh, [s], time_scale=ts, method="poisson")
    save_mesh_snapshot(vertices, faces, d,
                       os.path.join(OUT, f"10_geodesic_src{s}.html"),
                       title=f"Geodesic from vertex {s}", colormap="jet")

print("\nDone.")
