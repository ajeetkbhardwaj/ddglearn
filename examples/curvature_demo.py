"""Curvature & Shape Operator Demonstration."""

import sys
import numpy as np
from pathlib import Path

# Ensure the parent directory is in the path so 'import ddg' resolves correctly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ddg import (
    HalfEdgeMesh, load_mesh, gaussian_curvature, mean_curvature, 
    principal_curvatures, batch_visualization
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
    print(f"Mesh loaded: {mesh.n_vertices} vertices, {mesh.n_faces} faces.")

    print("Computing Gaussian Curvature...")
    K = gaussian_curvature(mesh)
    
    print("Computing Mean Curvature...")
    H = mean_curvature(mesh)
    
    print("Computing Principal Curvatures...")
    k1, k2 = principal_curvatures(mesh)

    try:
        import polyscope
        print("Launching Polyscope 3D Viewer...")
        batch_visualization(mesh, {
            "Gaussian Curvature (K)": ("scalar", K, {"cmap": "coolwarm"}),
            "Mean Curvature (H)": ("scalar", H, {"cmap": "viridis"}),
        })
    except ImportError:
        print("\nResults computed successfully! Install 'polyscope' to visualize.")

if __name__ == "__main__":
    main()