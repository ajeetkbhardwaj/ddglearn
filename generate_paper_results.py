import time
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D

# Import DDG library components
from core import HalfEdgeMesh
from core.mesh_io import load_mesh
from pde.geodesics import geodesic_distance
from spectral.wks import compute_wks
from spectral.eigen import eigen_decomposition

# Create output directory
OUT_DIR = "paper_results"
os.makedirs(OUT_DIR, exist_ok=True)

def generate_mesh_grid(nx, ny):
    """Generate a simple flat grid mesh for testing without degenerate faces."""
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    x, y = np.meshgrid(x, y)
    z = np.zeros_like(x)
    
    vertices = np.vstack([x.flatten(), y.flatten(), z.flatten()]).T
    faces = []
    
    for i in range(ny - 1):
        for j in range(nx - 1):
            p1 = i * nx + j
            p2 = p1 + 1
            p3 = (i + 1) * nx + j
            p4 = p3 + 1
            # Two triangles per grid cell
            faces.append([p1, p2, p3])
            faces.append([p2, p4, p3])
            
    return vertices, np.array(faces)

def plot_3d_mesh(vertices, faces, scalar_field, title, filename, cmap='viridis'):
    """Helper to save a 3D plot of a scalar field on a mesh."""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    
    surf = ax.plot_trisurf(x, y, faces, z, cmap=cmap, linewidth=0.2, antialiased=True)
    surf.set_array(scalar_field)
    
    ax.set_title(title)
    ax.axis('off')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {filename}")

def run_benchmarks():
    """Generate Table 1: Initialization Complexity."""
    print("Running Mesh Initialization Benchmarks...")
    
    resolutions = [(20, 20), (50, 50), (100, 100), (200, 200)]
    
    with open(os.path.join(OUT_DIR, "benchmark_table.csv"), "w") as f:
        f.write("Vertices,Faces,Initialization_Time_Seconds\n")
        
        for nx, ny in resolutions:
            V, F = generate_mesh_grid(nx, ny)
            
            start_time = time.time()
            mesh = HalfEdgeMesh(V, F)
            end_time = time.time()
            
            dur = end_time - start_time
            f.write(f"{len(V)},{len(F)},{dur:.5f}\n")
            print(f"Mesh: {len(V)} V, {len(F)} F -> {dur:.5f} sec")
            
    print("Saved benchmark_table.csv\n")

def generate_geodesics():
    """Generate Heat Method Geodesics Image."""
    print("Computing Geodesics (Heat Method)...")
    V, F = load_mesh("data/bunny.obj")
    mesh = HalfEdgeMesh(V, F)
    
    # Calculate distance from vertex 0
    dist = geodesic_distance(mesh, source_indices=[0], method="poisson")
    
    plot_3d_mesh(
        mesh.vertices, 
        mesh.faces, 
        dist, 
        "Heat Method Geodesic Distance", 
        "fig1_geodesics.png", 
        cmap="magma"
    )

def generate_laplacian_eigenfunctions():
    """Generate Laplacian Eigenfunctions Image."""
    print("Computing Laplacian Eigenfunctions...")
    V, F = load_mesh("data/bunny.obj")
    mesh = HalfEdgeMesh(V, F)
    
    # Get first 5 eigenvalues/eigenvectors
    vals, vecs = eigen_decomposition(mesh, k=5)
    
    # Plot the 2nd eigenfunction (1st non-trivial)
    phi_2 = vecs[:, 1]
    
    plot_3d_mesh(
        mesh.vertices, 
        mesh.faces, 
        phi_2, 
        "2nd Laplacian Eigenfunction (Fiedler Vector)", 
        "fig2_eigenfunction.png", 
        cmap="coolwarm"
    )

def generate_wks():
    """Generate WKS Signatures Image (2D plot)."""
    print("Computing Wave Kernel Signature (WKS)...")
    V, F = load_mesh("data/bunny.obj")
    mesh = HalfEdgeMesh(V, F)
    
    energies, wks = compute_wks(mesh, k=10)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Plot WKS for 3 distinct vertices
    vertices_to_plot = [0, mesh.n_vertices // 2, mesh.n_vertices - 1]
    
    for v in vertices_to_plot:
        ax.plot(energies, wks[v, :], label=f'Vertex {v}', linewidth=2)
        
    ax.set_title("Wave Kernel Signature (WKS) at Selected Vertices")
    ax.set_xlabel("Log-Energy Scale")
    ax.set_ylabel("WKS Value")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    filename = "fig3_wks.png"
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {filename}\n")

if __name__ == "__main__":
    print("=== DDG Library Results Generator ===")
    run_benchmarks()
    
    try:
        generate_geodesics()
    except Exception as e:
        print(f"Skipping Geodesics due to SciPy environment issue: {e}")
        
    try:
        generate_laplacian_eigenfunctions()
    except Exception as e:
        print(f"Skipping Eigenfunctions due to SciPy environment issue: {e}")
        
    try:
        generate_wks()
    except Exception as e:
        print(f"Skipping WKS due to SciPy environment issue: {e}")
        
    print(f"All generated files are saved in the '{OUT_DIR}' directory.")
