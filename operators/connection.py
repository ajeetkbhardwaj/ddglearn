"""Discrete Connection Laplacian and Parallel Transport.

Implements the Discrete Connection Laplacian for tangent vector fields
and tools for parallel transporting vectors across the mesh.
"""
import numpy as np

try:
    from scipy.sparse import coo_matrix
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def compute_vertex_bases(mesh) -> np.ndarray:
    """Compute an orthogonal tangent basis (X, Y) for each vertex.
    
    Returns:
        bases: (n_vertices, 2, 3) array where bases[i, 0] is X_i and bases[i, 1] is Y_i.
    """
    V = mesh.vertices
    F = mesh.faces
    n_v = mesh.n_vertices
    
    from core.performance import vertex_normals
    N = vertex_normals(V, F, normalize=True)

    bases = np.zeros((n_v, 2, 3), dtype=float)
    
    for i in range(n_v):
        n = N[i]
        # Pick a reference vector not strictly parallel to normal
        if abs(n[0]) < 0.9:
            ref = np.array([1.0, 0.0, 0.0])
        else:
            ref = np.array([0.0, 1.0, 0.0])
            
        x = ref - np.dot(ref, n) * n
        x /= np.linalg.norm(x)
        y = np.cross(n, x)
        
        bases[i, 0] = x
        bases[i, 1] = y
        
    return bases


def connection_laplacian(mesh):
    """Construct the Connection Laplacian matrix for tangent vector fields.
    
    Returns:
        L_conn: (2|V|, 2|V|) sparse matrix (Connection Laplacian)
        bases: (n_vertices, 2, 3) local tangent bases
    """
    V = mesh.vertices
    F = mesh.faces
    n_v = mesh.n_vertices
    
    bases = compute_vertex_bases(mesh)
    
    from core.performance import vertex_normals, cotangent_weights
    N = vertex_normals(V, F, normalize=True)

    weights_dict = cotangent_weights(V, F)
    weights_dict = {k: float(v) for k, v in weights_dict.items()}
        
    I_idx, J_idx, vals = [], [], []
    diag_sums = np.zeros(n_v)
    
    for (i, j), w in weights_dict.items():
        ni, nj = N[i], N[j]
        
        # Parallel transport X_i to j by rotating around ni x nj
        axis = np.cross(ni, nj)
        sin_theta = np.linalg.norm(axis)
        cos_theta = np.dot(ni, nj)
        
        if sin_theta > 1e-8:
            axis /= sin_theta
            xi = bases[i, 0]
            # Rodrigues' rotation formula
            xi_rot = xi * cos_theta + np.cross(axis, xi) * sin_theta + axis * np.dot(axis, xi) * (1 - cos_theta)
        else:
            xi_rot = bases[i, 0]
            
        xj = bases[j, 0]
        yj = bases[j, 1]
        
        cos_ij = np.dot(xi_rot, xj)
        sin_ij = np.dot(xi_rot, yj)
        
        R_ij = np.array([
            [cos_ij, -sin_ij],
            [sin_ij,  cos_ij]
        ])
        R_ji = R_ij.T
        
        # Add Off-Diagonal Block L_ij = -w * R_{j->i}
        for r in range(2):
            for c in range(2):
                I_idx.append(2*i + r)
                J_idx.append(2*j + c)
                vals.append(-w * R_ji[r, c])
                
                I_idx.append(2*j + r)
                J_idx.append(2*i + c)
                vals.append(-w * R_ij[r, c])
                
        diag_sums[i] += w
        diag_sums[j] += w
        
    # Add Block Diagonals L_ii
    for i in range(n_v):
        w = diag_sums[i]
        I_idx.extend([2*i, 2*i+1])
        J_idx.extend([2*i, 2*i+1])
        vals.extend([w, w])
        
    if _HAS_SCIPY:
        L_conn = coo_matrix((vals, (I_idx, J_idx)), shape=(2*n_v, 2*n_v)).tocsr()
    else:
        L_conn = np.zeros((2*n_v, 2*n_v))
        for r, c, v in zip(I_idx, J_idx, vals):
            L_conn[r, c] += v
            
    return L_conn, bases


def vertex_holonomy(mesh):
    """Compute the holonomy (angle defect) around each vertex.
    
    By the Gauss-Bonnet theorem, the discrete holonomy around a vertex 
    loop equates to its discrete Gaussian curvature integrated over area.
    """
    from geometry.curvature import gaussian_curvature
    K = gaussian_curvature(mesh)
    A = mesh.vertex_area_voronoi()

    return K * A

__all__ = ["connection_laplacian", "compute_vertex_bases", "vertex_holonomy"]