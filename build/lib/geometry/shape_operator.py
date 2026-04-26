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
    F = mesh.faces
    n_v = mesh.n_vertices

    normals = np.zeros((n_v, 3), dtype=float)

    for f in F:
        i, j, k = map(int, f)
        v0, v1, v2 = V[i], V[j], V[k]

        n = np.cross(v1 - v0, v2 - v0)
        area2 = np.linalg.norm(n)

        if area2 > 1e-14:
            normals[i] += n
            normals[j] += n
            normals[k] += n

    norms = np.linalg.norm(normals, axis=1)
    mask = norms > 1e-14
    normals[mask] /= norms[mask][:, None]

    return normals


# ------------------------------------------------------------
# Shape Operator Tensor (per vertex 2x2)
# ------------------------------------------------------------

def shape_operator_tensor(mesh):
    """
    Build symmetric 3x3 curvature tensor per vertex
    using edge-based cotangent weights.
    """
    V = mesh.vertices
    n_v = mesh.n_vertices
    normals = compute_vertex_normals(mesh)

    tensors = []

    for i in range(n_v):
        T = np.zeros((3, 3))

        neighbors = mesh.vertex_neighbors(i)

        xi = V[i]

        for j in neighbors:
            xj = V[j]
            e = xj - xi
            norm_e = np.linalg.norm(e)

            if norm_e < 1e-14:
                continue

            e /= norm_e
            T += np.outer(e, e)

        # Project to tangent plane
        n = normals[i]
        P = np.eye(3) - np.outer(n, n)
        S = P @ T @ P

        tensors.append(S)

    return tensors

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