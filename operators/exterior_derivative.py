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
    # Check if cached on mesh without triggering properties
    cache = getattr(mesh, "_cache", {})
    if "edge_list" in cache and "edge_map" in cache:
        return cache["edge_list"], cache["edge_map"]

    edge_map = {}
    edges = []
    processed = set()
    for he in mesh.halfedges:
        a = int(he.origin)
        b = int(mesh.halfedges[he.next].origin)
        key = (min(a, b), max(a, b))
        if key not in processed:
            processed.add(key)
            edge_map[key] = len(edges)
            edges.append((a, b))
    return edges, edge_map


def d0(mesh):
    """Return exterior derivative d0: vertices -> edges.

    Matrix shape: (n_edges, n_vertices)
    For each oriented edge (u->v) the row has -1 at u and +1 at v.
    """
    # Check cache
    cache = getattr(mesh, "_cache", None)
    if isinstance(cache, dict) and "d0" in cache:
        return cache["d0"]
    # Check mesh cache
    if hasattr(mesh, "_cache") and hasattr(mesh._cache, "operators"):
        if "d0" in mesh._cache.operators:
            return mesh._cache.operators["d0"]

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
        res = coo_matrix((data, (rows, cols)), shape=(n_e, n_v))
    else:
        M = np.zeros((n_e, n_v), dtype=float)
        for r, c, d in zip(rows, cols, data):
            M[r, c] = d
        res = M

    # Store in cache
    if isinstance(cache, dict):
        cache["d0"] = res
    elif hasattr(mesh, "_cache") and hasattr(mesh._cache, "operators"):
        mesh._cache.operators["d0"] = res
        
    return res


def d1(mesh):
    """Return exterior derivative d1: edges -> faces.

    Matrix shape: (n_faces, n_edges)
    Each face row contains +1/-1 for its boundary edges depending on
    orientation.
    """
    # Check cache
    cache = getattr(mesh, "_cache", None)
    if isinstance(cache, dict) and "d1" in cache:
        return cache["d1"]
    if hasattr(mesh, "_cache") and hasattr(mesh._cache, "operators"):
        if "d1" in mesh._cache.operators:
            return mesh._cache.operators["d1"]

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
        for a, b in face_edges:
            key = (min(a, b), max(a, b))
            ei = edge_map[key]
            # sign is +1 if face orientation matches stored orientation
            stored_a, stored_b = edges[ei]
            sign = 1.0 if (a == stored_a and b == stored_b) else -1.0
            rows.append(fi)
            cols.append(ei)
            data.append(sign)

    if _HAS_SCIPY:
        res = coo_matrix((data, (rows, cols)), shape=(n_f, n_e))
    else:
        M = np.zeros((n_f, n_e), dtype=float)
        for r, c, d in zip(rows, cols, data):
            M[r, c] = d
        res = M

    # Store in cache
    if isinstance(cache, dict):
        cache["d1"] = res
    elif hasattr(mesh, "_cache") and hasattr(mesh._cache, "operators"):
        mesh._cache.operators["d1"] = res
        
    return res


__all__ = ["d0", "d1"]
