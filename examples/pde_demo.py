"""PDE Solvers & Geodesics Demonstration."""

import sys
import numpy as np
from pathlib import Path

# Ensure the parent directory is in the path so 'import ddg' resolves correctly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ddg import HalfEdgeMesh, load_mesh, geodesic_distance, implicit_heat, batch_visualization

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
    
    # 1. Geodesic Distance via The Heat Method
    source_vertex = 0
    print(f"Computing Geodesic Distance from vertex {source_vertex}...")
    distances = geodesic_distance(mesh, source_indices=[source_vertex], method="poisson")
    
    # 2. Heat Equation Diffusion
    print("Simulating Heat Diffusion...")
    u0 = np.zeros(mesh.n_vertices)
    u0[source_vertex] = 100.0  # Hot spot
    u_diffused = implicit_heat(mesh, u0, t=0.01, steps=5)

    # Visualization
    try:
        import polyscope
        print("Launching Polyscope 3D Viewer...")
        batch_visualization(mesh, {
            "Geodesic Distance": ("scalar", distances, {"cmap": "viridis"}),
            "Heat Diffusion (t=0.05)": ("scalar", u_diffused, {"cmap": "inferno"}),
        })
    except ImportError:
        print("\nResults computed successfully!")
        print(f"Max distance: {distances.max():.4f}")
        print("Install 'polyscope' to visualize.")

if __name__ == "__main__":
    main()