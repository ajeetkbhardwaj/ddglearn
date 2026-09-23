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
    
    from ..core.performance import vertex_normals
    N = vertex_normals(V, F, normalize=True)

    bases = np.zeros((n_v, 2, 3), dtype=float)

    # Pick a reference vector per vertex that is not parallel to the normal.
    ref = np.where(np.abs(N[:, 0:1]) < 0.9, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])

    x = ref - np.sum(ref * N, axis=1, keepdims=True) * N
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    y = np.cross(N, x)

    bases[:, 0] = x
    bases[:, 1] = y

    return bases


def connection_laplacian(mesh):
    """Construct the Connection Laplacian matrix for tangent vector fields.

    Args:
        mesh (HalfEdgeMesh): The mesh instance.

    Returns:
        L_conn (scipy.sparse matrix): (2|V|, 2|V|) Connection Laplacian.
        bases (np.ndarray): (n_vertices, 2, 3) local tangent bases.
    """
    V = mesh.vertices
    F = mesh.faces
    n_v = mesh.n_vertices
    
    bases = compute_vertex_bases(mesh)
    
    from ..core.performance import vertex_normals, cotangent_weights
    N = vertex_normals(V, F, normalize=True)

    weights_dict = cotangent_weights(V, F)

    # Convert dict to arrays for vectorized operations.
    edges = np.array(list(weights_dict.keys()), dtype=int)
    if edges.size == 0:
        # No edges — return empty (degenerate) matrix.
        if _HAS_SCIPY:
            L_conn = coo_matrix((2 * n_v, 2 * n_v)).tocsr()
        else:
            L_conn = np.zeros((2 * n_v, 2 * n_v))
        return L_conn, bases

    ei = edges[:, 0]
    ej = edges[:, 1]
    w = np.array([weights_dict[k] for k in weights_dict.keys()], dtype=float)

    # Normals and bases at edge endpoints.
    ni, nj = N[ei], N[ej]
    xi = bases[ei, 0]  # (E, 3)
    xj = bases[ej, 0]
    yj = bases[ej, 1]

    # Rodrigues rotation of xi to j's tangent frame.
    axis = np.cross(ni, nj)  # (E, 3)
    sin_theta = np.linalg.norm(axis, axis=1)  # (E,)
    cos_theta = np.sum(ni * nj, axis=1)  # (E,)
    need_rot = sin_theta > 1e-8  # (E,)

    # Normalised rotation axis where needed.
    k = np.zeros_like(axis)
    k[need_rot] = axis[need_rot] / sin_theta[need_rot][:, None]

    cross_kxi = np.cross(k, xi)  # (E, 3)
    dot_kxi = np.sum(k * xi, axis=1, keepdims=True)  # (E, 1)

    xi_rot = (xi * cos_theta[:, None]
              + cross_kxi * sin_theta[:, None]
              + k * dot_kxi * (1 - cos_theta[:, None]))
    xi_rot[~need_rot] = xi[~need_rot]

    # Rotation coefficients.
    cos_ij = np.sum(xi_rot * xj, axis=1)  # (E,)
    sin_ij = np.sum(xi_rot * yj, axis=1)  # (E,)

    # Off-diagonal sparse entries (8 per edge).
    two_ei = 2 * ei
    two_ej = 2 * ej
    wc = w * cos_ij  # (E,)
    ws = w * sin_ij  # (E,)

    r_off = np.concatenate([
        two_ei, two_ei, two_ei + 1, two_ei + 1,
        two_ej, two_ej, two_ej + 1, two_ej + 1,
    ])
    c_off = np.concatenate([
        two_ej, two_ej + 1, two_ej, two_ej + 1,
        two_ei, two_ei + 1, two_ei, two_ei + 1,
    ])
    v_off = np.concatenate([
        -wc, -ws, ws, -wc,
        -wc, ws, -ws, -wc,
    ])

    # Diagonal: sum of incident cotangent weights per vertex.
    diag = np.zeros(n_v, dtype=float)
    np.add.at(diag, ei, w)
    np.add.at(diag, ej, w)

    r_diag = np.concatenate([2 * np.arange(n_v), 2 * np.arange(n_v) + 1])
    c_diag = r_diag.copy()
    v_diag = np.concatenate([diag, diag])

    # Assemble.
    I_idx = np.concatenate([r_off, r_diag])
    J_idx = np.concatenate([c_off, c_diag])
    vals = np.concatenate([v_off, v_diag])

    if _HAS_SCIPY:
        L_conn = coo_matrix((vals, (I_idx, J_idx)), shape=(2 * n_v, 2 * n_v)).tocsr()
    else:
        L_conn = np.zeros((2 * n_v, 2 * n_v))
        np.add.at(L_conn, (I_idx, J_idx), vals)

    return L_conn, bases


def vertex_holonomy(mesh):
    """Compute the holonomy (angle defect) around each vertex.
    
    By the Gauss-Bonnet theorem, the discrete holonomy around a vertex 
    loop equates to its discrete Gaussian curvature integrated over area.
    """
    from ..geometry.curvature import gaussian_curvature
    K = gaussian_curvature(mesh)
    A = mesh.vertex_area_voronoi()

    return K * A

__all__ = ["connection_laplacian", "compute_vertex_bases", "vertex_holonomy"]