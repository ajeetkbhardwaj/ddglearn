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
    if hasattr(mesh, "vertex_area_voronoi_tensor"):
        a = mesh.vertex_area_voronoi_tensor
    else:
        a = mesh.vertex_area_voronoi()
        
    use_torch = _HAS_TORCH and hasattr(a, "device") and isinstance(a, torch.Tensor)
    
    if use_torch:
        idx = torch.arange(len(a), device=a.device)
        return torch.sparse_coo_tensor(torch.stack([idx, idx]), a, (len(a), len(a))).coalesce()
        
    if _HAS_SCIPY:
        return diags(a)
    else:
        return np.diag(a)


def hodge_star_2(mesh):
    """Return *2 as a diagonal matrix (face areas).

    Shape: (n_faces, n_faces)
    """
    if hasattr(mesh, "face_areas_tensor"):
        a = mesh.face_areas_tensor
    else:
        a = _face_areas(mesh)
        
    use_torch = _HAS_TORCH and hasattr(a, "device") and isinstance(a, torch.Tensor)
    
    if use_torch:
        idx = torch.arange(len(a), device=a.device)
        return torch.sparse_coo_tensor(torch.stack([idx, idx]), a, (len(a), len(a))).coalesce()
        
    if _HAS_SCIPY:
        return diags(a)
    else:
        return np.diag(a)


def hodge_star_1(mesh):
    """Return *1 as a diagonal matrix (cotangent weights).

    Shape: (n_edges, n_edges)
    """
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

        F_cpu = F.cpu().numpy()
        ei = np.array([edge_map[(min(int(j), int(k)), max(int(j), int(k)))] for j, k in zip(F_cpu[:, 1], F_cpu[:, 2])])
        ej = np.array([edge_map[(min(int(i), int(k)), max(int(i), int(k)))] for i, k in zip(F_cpu[:, 0], F_cpu[:, 2])])
        ek = np.array([edge_map[(min(int(i), int(j)), max(int(i), int(j)))] for i, j in zip(F_cpu[:, 0], F_cpu[:, 1])])
        
        weights = torch.zeros(n_e, dtype=V.dtype, device=V.device)
        weights.scatter_add_(0, torch.tensor(ei, device=V.device, dtype=torch.long), cot0)
        weights.scatter_add_(0, torch.tensor(ej, device=V.device, dtype=torch.long), cot1)
        weights.scatter_add_(0, torch.tensor(ek, device=V.device, dtype=torch.long), cot2)
        
        weights = torch.clamp(weights, min=1e-12)
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

        ei = np.array([edge_map[(min(int(j), int(k)), max(int(j), int(k)))] for j, k in zip(F[:, 1], F[:, 2])])
        ej = np.array([edge_map[(min(int(i), int(k)), max(int(i), int(k)))] for i, k in zip(F[:, 0], F[:, 2])])
        ek = np.array([edge_map[(min(int(i), int(j)), max(int(i), int(j)))] for i, j in zip(F[:, 0], F[:, 1])])
        
        weights = np.zeros(n_e)
        np.add.at(weights, ei, cot0)
        np.add.at(weights, ej, cot1)
        np.add.at(weights, ek, cot2)
        
        weights = np.maximum(weights, 1e-12)

    if _HAS_SCIPY:
        return diags(weights)
    else:
        return np.diag(weights)

__all__ = ["hodge_star_0", "hodge_star_1", "hodge_star_2"]
