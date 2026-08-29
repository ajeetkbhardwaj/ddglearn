"""Hodge star operators for 0-, 1-, and 2-forms.

These implementations use simple primal areas for 0- and 2-forms and a
practical approximation for 1-forms which is sufficient for small tests.
"""
import numpy as np
from typing import Tuple
from .exterior_derivative import edge_list_and_map

try:
    from scipy.sparse import diags
    _HAS_SCIPY = True
except Exception:
    diags = None
    _HAS_SCIPY = False


def _face_areas(mesh):
    V, F = mesh.vertices, mesh.faces
    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    return 0.5 * np.linalg.norm(cross, axis=1)


def hodge_star_0(mesh):
    """Return *0 as a diagonal matrix (vertex areas).

    Shape: (n_vertices, n_vertices)
    """
    cache = getattr(mesh, "_cache", None)
    if isinstance(cache, dict) and "hodge_0" in cache:
        return cache["hodge_0"]
    if hasattr(mesh, "_cache") and hasattr(mesh._cache, "operators"):
        if "hodge_0" in mesh._cache.operators:
            return mesh._cache.operators["hodge_0"]

    if hasattr(mesh, "vertex_area_voronoi_tensor"):
        a = mesh.vertex_area_voronoi_tensor
    else:
        a = mesh.vertex_area_voronoi()

    if _HAS_SCIPY:
        res = diags(a)
    else:
        res = np.diag(a)

    if isinstance(cache, dict):
        cache["hodge_0"] = res
    elif hasattr(mesh, "_cache") and hasattr(mesh._cache, "operators"):
        mesh._cache.operators["hodge_0"] = res
    return res


def hodge_star_2(mesh):
    """Return *2 as a diagonal matrix (face areas).

    Shape: (n_faces, n_faces)
    """
    cache = getattr(mesh, "_cache", None)
    if isinstance(cache, dict) and "hodge_2" in cache:
        return cache["hodge_2"]
    if hasattr(mesh, "_cache") and hasattr(mesh._cache, "operators"):
        if "hodge_2" in mesh._cache.operators:
            return mesh._cache.operators["hodge_2"]

    if hasattr(mesh, "face_areas_tensor"):
        a = mesh.face_areas_tensor
    else:
        a = _face_areas(mesh)

    if _HAS_SCIPY:
        res = diags(a)
    else:
        res = np.diag(a)

    if isinstance(cache, dict):
        cache["hodge_2"] = res
    elif hasattr(mesh, "_cache") and hasattr(mesh._cache, "operators"):
        mesh._cache.operators["hodge_2"] = res
    return res


def hodge_star_1(mesh):
    """Return *1 as a diagonal matrix (cotangent weights).

    Shape: (n_edges, n_edges)
    """
    cache = getattr(mesh, "_cache", None)
    if isinstance(cache, dict) and "hodge_1" in cache:
        return cache["hodge_1"]
    if hasattr(mesh, "_cache") and hasattr(mesh._cache, "operators"):
        if "hodge_1" in mesh._cache.operators:
            return mesh._cache.operators["hodge_1"]

    edges, edge_map = edge_list_and_map(mesh)
    n_e = len(edges)

    V = mesh.vertices
    F = mesh.faces

    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    e0, e1, e2 = v2 - v1, v0 - v2, v1 - v0

    cross0, cross1, cross2 = np.cross(e1, -e2), np.cross(e2, -e0), np.cross(e0, -e1)
    norm0 = np.maximum(np.linalg.norm(cross0, axis=1), 1e-12)
    norm1 = np.maximum(np.linalg.norm(cross1, axis=1), 1e-12)
    norm2 = np.maximum(np.linalg.norm(cross2, axis=1), 1e-12)

    cot0 = 0.5 * np.sum(e1 * (-e2), axis=1) / norm0
    cot1 = 0.5 * np.sum(e2 * (-e0), axis=1) / norm1
    cot2 = 0.5 * np.sum(e0 * (-e1), axis=1) / norm2

    edges_np = np.array(edges, dtype=np.int64)
    n_v_np = np.int64(len(V))
    edge_hash = np.minimum(edges_np[:, 0], edges_np[:, 1]) * n_v_np + np.maximum(edges_np[:, 0], edges_np[:, 1])
    sorted_idx = np.argsort(edge_hash)
    sorted_hash = edge_hash[sorted_idx]

    def get_edge_idx_np(a, b):
        h = np.minimum(a, b).astype(np.int64) * n_v_np + np.maximum(a, b).astype(np.int64)
        return sorted_idx[np.searchsorted(sorted_hash, h)]

    ei = get_edge_idx_np(F[:, 1], F[:, 2])
    ej = get_edge_idx_np(F[:, 0], F[:, 2])
    ek = get_edge_idx_np(F[:, 0], F[:, 1])

    weights = np.zeros(n_e)
    np.add.at(weights, ei, cot0)
    np.add.at(weights, ej, cot1)
    np.add.at(weights, ek, cot2)

    if _HAS_SCIPY:
        res = diags(weights)
    else:
        res = np.diag(weights)

    if isinstance(cache, dict):
        cache["hodge_1"] = res
    elif hasattr(mesh, "_cache") and hasattr(mesh._cache, "operators"):
        mesh._cache.operators["hodge_1"] = res
    return res

__all__ = ["hodge_star_0", "hodge_star_1", "hodge_star_2"]
