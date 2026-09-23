"""Mesh parameterization algorithms.

Maps 3D surfaces with disk topology to 2D domains (UV mapping) 
by solving elliptic PDEs with boundary constraints.
"""
import numpy as np

def harmonic_parameterization(mesh) -> np.ndarray:
    r"""Compute a 2D harmonic parameterization of a disk-topology mesh.
    
    Maps the topological boundary of the mesh to a unit circle, and 
    solves the Laplace equation (\Delta u = 0) for interior coordinates.
    This naturally leverages the cotangent Laplacian for minimal angular distortion.
    
    Args:
        mesh (HalfEdgeMesh): The mesh instance.
        
    Returns:
        uv_coords (np.ndarray): (n_vertices, 2) array of 2D UV coordinates.
    """
    from ..pde.poisson import solve_poisson
    
    boundary_edges = mesh.boundary_edges()
    if not boundary_edges:
        raise ValueError("Mesh has no boundary. Cannot compute harmonic parameterization on a closed surface.")
        
    # 1. Build a directed connectivity graph of the boundary
    b_adj = {}
    for u, v in boundary_edges:
        b_adj[u] = v
        
    # 2. Traverse the boundary loop in order
    start_v = boundary_edges[0][0]
    loop = [start_v]
    curr = b_adj[start_v]
    
    while curr != start_v:
        loop.append(curr)
        if curr not in b_adj:
            raise ValueError("Inconsistent boundary topology.")
        curr = b_adj[curr]
        if len(loop) > mesh.n_vertices:
            raise ValueError("Multiple boundary holes detected; standard Tutte requires a single disk boundary.")
            
    # 3. Map boundary loop to the 2D unit circle
    n_b = len(loop)
    angles = np.linspace(0, 2 * np.pi, n_b, endpoint=False)
    circle_x = np.cos(angles)
    circle_y = np.sin(angles)
    
    dirichlet_idx = np.array(loop)
    
    # 4. Solve the Laplace equation (\Delta u = 0) for interior coordinates
    f_zero = np.zeros(mesh.n_vertices)
    
    u_coords = solve_poisson(mesh, f_zero, dirichlet_indices=dirichlet_idx, dirichlet_values=circle_x)
    v_coords = solve_poisson(mesh, f_zero, dirichlet_indices=dirichlet_idx, dirichlet_values=circle_y)
    
    return np.column_stack((u_coords, v_coords))

__all__ = ["harmonic_parameterization"]