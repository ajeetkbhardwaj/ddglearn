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

    edges_np = np.array(edges, dtype=int)  # (n_e, 2)
    rows = np.repeat(np.arange(n_e), 2)
    cols = edges_np.ravel()
    data = np.tile([-1.0, 1.0], n_e)

    if _HAS_SCIPY:
        res = coo_matrix((data, (rows, cols)), shape=(n_e, n_v))
    else:
        M = np.zeros((n_e, n_v), dtype=float)
        np.add.at(M, (rows, cols), data)
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

    F = np.asarray(mesh.faces, dtype=int)  # (n_f, 3)
    edges_np = np.array(edges, dtype=int)  # (n_e, 2)

    # Build hash for fast edge lookup: key = min * n_v + max
    n_v = mesh.n_vertices if hasattr(mesh, "n_vertices") else int(edges_np.max()) + 1
    edge_hash = np.minimum(edges_np[:, 0], edges_np[:, 1]).astype(np.int64) * n_v + \
                np.maximum(edges_np[:, 0], edges_np[:, 1]).astype(np.int64)
    sorted_idx = np.argsort(edge_hash)
    sorted_hash = edge_hash[sorted_idx]

    def lookup_edge_idx(a, b):
        h = np.minimum(a, b).astype(np.int64) * n_v + np.maximum(a, b).astype(np.int64)
        return sorted_idx[np.searchsorted(sorted_hash, h)]

    # Face edges (3 per face) and orientation signs
    face_edges_a = np.column_stack([F[:, 0], F[:, 1], F[:, 2]])  # (n_f, 3)
    face_edges_b = np.column_stack([F[:, 1], F[:, 2], F[:, 0]])  # (n_f, 3)
    ei = lookup_edge_idx(face_edges_a.ravel(), face_edges_b.ravel())  # (n_f * 3,)

    # Sign: +1 if face orientation matches stored edge orientation, else -1
    stored_a = edges_np[ei, 0]
    stored_b = edges_np[ei, 1]
    match = (face_edges_a.ravel() == stored_a) & (face_edges_b.ravel() == stored_b)
    signs = np.where(match, 1.0, -1.0)

    face_idx = np.repeat(np.arange(n_f), 3)

    if _HAS_SCIPY:
        res = coo_matrix((signs, (face_idx, ei)), shape=(n_f, n_e))
    else:
        M = np.zeros((n_f, n_e), dtype=float)
        np.add.at(M, (face_idx, ei), signs)
        res = M

    # Store in cache
    if isinstance(cache, dict):
        cache["d1"] = res
    elif hasattr(mesh, "_cache") and hasattr(mesh._cache, "operators"):
        mesh._cache.operators["d1"] = res
        
    return res


__all__ = ["d0", "d1"]
