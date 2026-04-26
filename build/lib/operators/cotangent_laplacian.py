"""Cotangent Laplacian operator.

Provides the discrete Laplace-Beltrami operator using the cotangent formula directly.
"""

import numpy as np

try:
    from scipy.sparse import coo_matrix
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def cotangent_laplacian(mesh):
    """Construct the cotangent Laplace-Beltrami operator matrix L.

    Args:
        mesh: The HalfEdgeMesh or TorchHalfEdgeMesh instance.

    Returns:
        L: (n_vertices, n_vertices) PyTorch sparse tensor (if GPU mesh), 
           SciPy sparse matrix (if installed), or dense numpy array.
    """
    V = mesh.vertices
    F = mesh.faces
    n_v = mesh.n_vertices

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
        
        I = torch.cat([F[:, 1], F[:, 2], F[:, 1], F[:, 2], F[:, 2], F[:, 0], F[:, 2], F[:, 0], F[:, 0], F[:, 1], F[:, 0], F[:, 1]]).long()
        J = torch.cat([F[:, 2], F[:, 1], F[:, 1], F[:, 2], F[:, 0], F[:, 2], F[:, 2], F[:, 0], F[:, 1], F[:, 0], F[:, 0], F[:, 1]]).long()
        vals = torch.cat([cot0, cot0, -cot0, -cot0, cot1, cot1, -cot1, -cot1, cot2, cot2, -cot2, -cot2])
                          
        return torch.sparse_coo_tensor(torch.stack([I, J]), vals, (n_v, n_v)).coalesce()

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
        
        I = np.concatenate([F[:, 1], F[:, 2], F[:, 1], F[:, 2], F[:, 2], F[:, 0], F[:, 2], F[:, 0], F[:, 0], F[:, 1], F[:, 0], F[:, 1]])
        J = np.concatenate([F[:, 2], F[:, 1], F[:, 1], F[:, 2], F[:, 0], F[:, 2], F[:, 2], F[:, 0], F[:, 1], F[:, 0], F[:, 0], F[:, 1]])
        vals = np.concatenate([cot0, cot0, -cot0, -cot0, cot1, cot1, -cot1, -cot1, cot2, cot2, -cot2, -cot2])

        if _HAS_SCIPY:
            return coo_matrix((vals, (I, J)), shape=(n_v, n_v)).tocsr()
        else:
            L = np.zeros((n_v, n_v), dtype=float)
            np.add.at(L, (I, J), vals)
            return L

__all__ = ["cotangent_laplacian"]