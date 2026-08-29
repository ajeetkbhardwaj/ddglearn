"""
Discrete Shape Operator (Weingarten Map) and Principal Directions.

Implements curvature tensor estimation via projection of the
mean curvature normal onto the tangent plane and construction
of a symmetric 2x2 tensor per vertex.

References:
- Meyer et al. 2002
- Crane DDG course notes
- Botsch et al., Polygon Mesh Processing
"""

import numpy as np

# ------------------------------------------------------------
# Vertex Normals
# ------------------------------------------------------------

def compute_vertex_normals(mesh) -> np.ndarray:
    V = mesh.vertices
    F = np.asarray(mesh.faces)
    n_v = mesh.n_vertices

    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    area2 = np.linalg.norm(n, axis=1)

    normals = np.zeros((n_v, 3), dtype=float)
    mask = area2 > 1e-14
    np.add.at(normals, F[mask, 0], n[mask])
    np.add.at(normals, F[mask, 1], n[mask])
    np.add.at(normals, F[mask, 2], n[mask])

    norms = np.linalg.norm(normals, axis=1)
    m = norms > 1e-14
    normals[m] /= norms[m][:, None]

    return normals


# ------------------------------------------------------------
# Shape Operator Tensor (per vertex 2x2)
# ------------------------------------------------------------

def shape_operator_tensor(mesh):
    """
    Build symmetric 3x3 curvature tensor per vertex
    using edge-based cotangent weights.

    Returns an (n_vertices, 3, 3) array of symmetric tensors.
    """
    V = np.asarray(mesh.vertices, dtype=float)
    F = np.asarray(mesh.faces)
    n_v = mesh.n_vertices
    normals = compute_vertex_normals(mesh)

    # Undirected edges (i, j) from all triangle edges.
    all_edges = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    lo = np.min(all_edges, axis=1)
    hi = np.max(all_edges, axis=1)
    uniq, inv = np.unique(np.stack([lo, hi], axis=1), axis=0, return_inverse=True)

    a = uniq[:, 0]
    b = uniq[:, 1]
    e = V[b] - V[a]
    nrm = np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-14)
    e = e / nrm
    outer = e[:, :, None] * e[:, None, :]  # (E, 3, 3)

    # Each undirected edge contributes e e^T to both endpoints.
    T = np.zeros((n_v, 3, 3), dtype=float)
    np.add.at(T, a, outer)
    np.add.at(T, b, outer)

    # Project each tensor to its tangent plane: S_i = P_i T_i P_i.
    P = np.eye(3)[None] - normals[:, :, None] * normals[:, None, :]  # (n_v, 3, 3)
    S = np.einsum("vik,vkl,vjl->vij", P, T, P)

    return S

# ------------------------------------------------------------
# Principal Directions
# ------------------------------------------------------------

def principal_directions(mesh):
    tensors = shape_operator_tensor(mesh)
    n_v = mesh.n_vertices

    k1_dirs = np.zeros((n_v, 3))
    k2_dirs = np.zeros((n_v, 3))

    for i, S in enumerate(tensors):

        vals, vecs = np.linalg.eigh(S)

        idx = np.argsort(vals)[::-1]
        vecs = vecs[:, idx]

        k1_dirs[i] = vecs[:, 0]
        k2_dirs[i] = vecs[:, 1]

    return k1_dirs, k2_dirs

__all__ = [
    "compute_vertex_normals",
    "shape_operator_tensor",
    "principal_directions",
]