"""Advanced Topologies & Research Algorithms Demonstration."""

import sys
import numpy as np
from pathlib import Path

# Ensure the parent directory is in the path so 'import ddg' resolves correctly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ddg import (
    HalfEdgeMesh, load_mesh, 
    fluid_velocity_from_vorticity, 
    decimate_mesh, 
    sinkhorn_wasserstein,
    batch_visualization
)

def get_demo_mesh():
    data_path = Path(__file__).resolve().parents[1] / "data" / "bunny.obj"
    if data_path.exists():
        print(f"Loading {data_path}...")
        V, F = load_mesh(str(data_path))
        return HalfEdgeMesh(V, F)
    else:
        print("Notice: data/bunny.obj not found. Falling back to synthetic octahedron.")
        V = np.array([[1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]], dtype=float)
        F = np.array([[0,4,2], [2,4,1], [1,4,3], [3,4,0], [0,2,5], [2,1,5], [1,3,5], [3,0,5]], dtype=int)
        return HalfEdgeMesh(V, F)

def main():
    mesh = get_demo_mesh()
    
    print("--- 1. Eulerian Fluid Simulation ---")
    # Create some random vorticity field on the surface
    omega = np.random.randn(mesh.n_vertices)
    print("Calculating incompressible fluid velocity (1-form flux) from vorticity...")
    flux, omega_diffused = fluid_velocity_from_vorticity(mesh, omega, viscosity=0.01, dt=0.1)
    print(f"Successfully computed {len(flux)} edge flux vectors!")

    print("\n--- 2. Mesh Decimation ---")
    print(f"Original Mesh: {mesh.n_vertices} vertices.")
    # Decimate the mesh using grid resolution (larger number = heavier decimation)
    decimated_mesh = decimate_mesh(mesh, grid_resolution=0.01)
    print(f"Decimated Mesh: {decimated_mesh.n_vertices} vertices.")

    print("\n--- 3. Optimal Transport (Sinkhorn Wasserstein) ---")
    p = np.zeros(mesh.n_vertices)
    q = np.zeros(mesh.n_vertices)
    p[0] = 1.0  # Mass at vertex 0
    q[-1] = 1.0 # Mass at the last vertex
    
    print("Computing optimal transport map between two Dirac deltas...")
    dist, u, v = sinkhorn_wasserstein(mesh, p, q, t=0.05, max_iter=20)
    print(f"Wasserstein distance mapped: {dist:.4f}")
    
    print("\nAdvanced research algorithms executed flawlessly!")

if __name__ == "__main__":
    main()