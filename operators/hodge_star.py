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

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def _face_areas(mesh):
    V, F = mesh.vertices, mesh.faces
    use_torch = _HAS_TORCH and hasattr(V, "device") and isinstance(V, torch.Tensor)
    
    if use_torch:
        v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        cross = torch.linalg.cross(v1 - v0, v2 - v0)
        return 0.5 * torch.linalg.norm(cross, dim=1)
    else:
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
        
    use_torch = _HAS_TORCH and hasattr(a, "device") and isinstance(a, torch.Tensor)
    
    if use_torch:
        idx = torch.arange(len(a), device=a.device)
        res = torch.sparse_coo_tensor(torch.stack([idx, idx]), a, (len(a), len(a))).coalesce()
    elif _HAS_SCIPY:
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
        
    use_torch = _HAS_TORCH and hasattr(a, "device") and isinstance(a, torch.Tensor)
    
    if use_torch:
        idx = torch.arange(len(a), device=a.device)
        res = torch.sparse_coo_tensor(torch.stack([idx, idx]), a, (len(a), len(a))).coalesce()
    elif _HAS_SCIPY:
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
    use_torch = _HAS_TORCH and hasattr(V, "device") and isinstance(V, torch.Tensor)

    if use_torch:
        v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        e0, e1, e2 = v2 - v1, v0 - v2, v1 - v0
        
        cross0, cross1, cross2 = torch.linalg.cross(e1, -e2), torch.linalg.cross(e2, -e0), torch.linalg.cross(e0, -e1)
        norm0 = torch.linalg.norm(cross0, dim=1).clamp(min=1e-12)
        norm1 = torch.linalg.norm(cross1, dim=1).clamp(min=1e-12)
        norm2 = torch.linalg.norm(cross2, dim=1).clamp(min=1e-12)
        
        cot0 = 0.5 * torch.sum(e1 * (-e2), dim=1) / norm0
        cot1 = 0.5 * torch.sum(e2 * (-e0), dim=1) / norm1
        cot2 = 0.5 * torch.sum(e0 * (-e1), dim=1) / norm2

        edges_t = torch.tensor(edges, dtype=torch.long, device=V.device)
        n_v_t = torch.tensor(len(V), dtype=torch.long, device=V.device)
        edge_hash = torch.minimum(edges_t[:, 0], edges_t[:, 1]) * n_v_t + torch.maximum(edges_t[:, 0], edges_t[:, 1])
        sorted_idx = torch.argsort(edge_hash)
        sorted_hash = edge_hash[sorted_idx]
        
        def get_edge_idx(a, b):
            h = torch.minimum(a, b).long() * n_v_t + torch.maximum(a, b).long()
            return sorted_idx[torch.searchsorted(sorted_hash, h)]
            
        ei = get_edge_idx(F[:, 1], F[:, 2])
        ej = get_edge_idx(F[:, 0], F[:, 2])
        ek = get_edge_idx(F[:, 0], F[:, 1])

        weights = torch.zeros(n_e, dtype=V.dtype, device=V.device)
        weights.scatter_add_(0, ei, cot0)
        weights.scatter_add_(0, ej, cot1)
        weights.scatter_add_(0, ek, cot2)
        
        idx = torch.arange(n_e, device=V.device)
        return torch.sparse_coo_tensor(torch.stack([idx, idx]), weights, (n_e, n_e)).coalesce()
    else:
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
