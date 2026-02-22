"""Hodge star operators for 0-, 1-, and 2-forms.

These implementations use simple primal areas for 0- and 2-forms and a
practical approximation for 1-forms which is sufficient for small tests.
"""
import numpy as np
from typing import Tuple

try:
    from scipy.sparse import diags
    _HAS_SCIPY = True
except Exception:
    diags = None
    _HAS_SCIPY = False


def _face_areas(mesh) -> np.ndarray:
    areas = np.zeros(mesh.n_faces, dtype=float)
    for fi, f in enumerate(mesh.faces):
        v0, v1, v2 = [mesh.vertices[int(i)] for i in f]
        areas[fi] = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
    return areas


def hodge_star_0(mesh):
    """Return *0 as a diagonal matrix (vertex areas).

    Shape: (n_vertices, n_vertices)
    """
    a = mesh.vertex_area_voronoi()
    if _HAS_SCIPY:
        return diags(a)
    else:
        return np.diag(a)


def hodge_star_2(mesh):
    """Return *2 as a diagonal matrix (face areas).

    Shape: (n_faces, n_faces)
    """
    a = _face_areas(mesh)
    if _HAS_SCIPY:
        return diags(a)
    else:
        return np.diag(a)


from .exterior_derivative import edge_list_and_map

def hodge_star_1(mesh):
    edges, edge_map = edge_list_and_map(mesh)
    n_e = len(edges)

    face_areas = _face_areas(mesh)

    # map edge → adjacent faces 
    edge_to_faces = {i: [] for i in range(n_e)}

    for fi, f in enumerate(mesh.faces):
        v0, v1, v2 = [int(x) for x in f]
        for a, b in [(v0, v1), (v1, v2), (v2, v0)]:
            key = (min(a, b), max(a, b))
            ei = edge_map[key]
            edge_to_faces[ei].append(fi)

    weights = np.zeros(n_e)

    eps = np.finfo(float).eps

    for ei, (a, b) in enumerate(edges):
        pa = mesh.vertices[a]
        pb = mesh.vertices[b]
        L = np.linalg.norm(pb - pa)

        adj = edge_to_faces[ei]
        if len(adj) == 0:
            dual_measure = 1.0
        else:
            dual_measure = np.mean(face_areas[adj])

        weights[ei] = L / max(dual_measure, eps)

    if _HAS_SCIPY:
        return diags(weights)
    else:
        return np.diag(weights)

__all__ = ["hodge_star_0", "hodge_star_1", "hodge_star_2"]
