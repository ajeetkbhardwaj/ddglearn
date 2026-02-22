"""Discrete exterior derivative matrices for triangle meshes.

Provides d0: C^0 -> C^1 and d1: C^1 -> C^2 using the `HalfEdgeMesh`.
Sparse matrices are returned when `scipy.sparse` is available, otherwise
dense numpy arrays are used (suitable for small meshes/tests).
"""
from typing import List, Tuple
import numpy as np

try:
    from scipy.sparse import coo_matrix
    _HAS_SCIPY = True
except Exception:
    coo_matrix = None
    _HAS_SCIPY = False


def edge_list_and_map(mesh) -> Tuple[List[Tuple[int, int]], dict]:
    edge_map = {}
    edges = []
    for he_idx, he in enumerate(mesh.halfedges):
        a = int(he.origin)
        b = int(mesh.halfedges[he.next].origin)
        key = (min(a, b), max(a, b))
        if key not in edge_map:
            # store oriented edge as (a,b) following this halfedge
            edge_map[key] = len(edges)
            edges.append((a, b))
    return edges, edge_map


def d0(mesh):
    """Return exterior derivative d0: vertices -> edges.

    Matrix shape: (n_edges, n_vertices)
    For each oriented edge (u->v) the row has -1 at u and +1 at v.
    """
    edges, edge_map = edge_list_and_map(mesh)
    n_e = len(edges)
    n_v = mesh.n_vertices

    rows = []
    cols = []
    data = []
    for ei, (u, v) in enumerate(edges):
        rows.extend([ei, ei])
        cols.extend([u, v])
        data.extend([-1.0, 1.0])

    if _HAS_SCIPY:
        return coo_matrix((data, (rows, cols)), shape=(n_e, n_v))
    else:
        M = np.zeros((n_e, n_v), dtype=float)
        for r, c, d in zip(rows, cols, data):
            M[r, c] = d
        return M


def d1(mesh):
    """Return exterior derivative d1: edges -> faces.

    Matrix shape: (n_faces, n_edges)
    Each face row contains +1/-1 for its boundary edges depending on
    orientation.
    """
    edges, edge_map = edge_list_and_map(mesh)
    n_e = len(edges)
    n_f = mesh.n_faces

    rows = []
    cols = []
    data = []
    for fi, face in enumerate(mesh.faces):
        # face vertices in order (assumed CCW)
        v0, v1, v2 = [int(x) for x in face]
        face_edges = [(v0, v1), (v1, v2), (v2, v0)]
        for (a, b) in face_edges:
            key = (min(a, b), max(a, b))
            ei = edge_map[key]
            # sign is +1 if face orientation matches stored orientation
            stored_a, stored_b = edges[ei]
            sign = 1.0 if (a == stored_a and b == stored_b) else -1.0
            rows.append(fi)
            cols.append(ei)
            data.append(sign)

    if _HAS_SCIPY:
        return coo_matrix((data, (rows, cols)), shape=(n_f, n_e))
    else:
        M = np.zeros((n_f, n_e), dtype=float)
        for r, c, d in zip(rows, cols, data):
            M[r, c] = d
        return M


__all__ = ["d0", "d1"]
