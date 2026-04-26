"""Spectral Geometry Demonstration (Laplace-Beltrami Eigenmodes & Shape Signatures)."""

import sys
import numpy as np
from pathlib import Path

# Ensure the parent directory is in the path so 'import ddg' resolves correctly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ddg import HalfEdgeMesh, load_mesh, eigen_decomposition, compute_hks, compute_wks, batch_visualization

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
    
    k = min(20, mesh.n_vertices - 1)
    print(f"Computing first {k} Eigenvalues and Eigenvectors of the Cotangent Laplacian...")
    evals, evecs = eigen_decomposition(mesh, k=k)
    
    print("Computing Heat Kernel Signatures (HKS)...")
    times, hks = compute_hks(mesh, k=k)
    
    print("Computing Wave Kernel Signatures (WKS)...")
    energies, wks = compute_wks(mesh, k=k)

    # Visualization
    try:
        import polyscope
        print("Launching Polyscope 3D Viewer...")
        batch_visualization(mesh, {
            "Laplacian Eigenmode 1": ("scalar", evecs[:, 1], {"cmap": "coolwarm"}),
            "Laplacian Eigenmode 2": ("scalar", evecs[:, 2], {"cmap": "coolwarm"}),
            "HKS (t-small)": ("scalar", hks[:, 1], {"cmap": "plasma", "vminmax": (np.min(hks[:, 1]), np.percentile(hks[:, 1], 98))}),
            "WKS (e-small)": ("scalar", wks[:, 1], {"cmap": "plasma", "vminmax": (np.min(wks[:, 1]), np.percentile(wks[:, 1], 98))}),
        })
    except ImportError:
        print("\nResults computed successfully!")
        print("Install 'polyscope' to view the harmonics on the mesh surface.")

if __name__ == "__main__":
    main()