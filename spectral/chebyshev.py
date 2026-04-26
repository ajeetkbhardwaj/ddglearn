"""Chebyshev polynomial spectral filters.

Allows for fast Spectral Graph Convolutions (Geometric Deep Learning)
without requiring full eigendecomposition of the mesh Laplacian.
"""
import numpy as np
from typing import Union, List

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from operators.cotangent_laplacian import cotangent_laplacian
from operators.hodge_star import hodge_star_0


def scaled_laplacian(mesh):
    """Compute the scaled symmetric normalized Laplacian.

    L_sym = M^{-1/2} L M^{-1/2}
    Where M is the mass matrix (hodge_star_0) and L is the cotangent laplacian.
    
    Returns an operator mapping bounded to [-1, 1].
    """
    L = cotangent_laplacian(mesh)
    M = hodge_star_0(mesh)
    
    use_torch = _HAS_TORCH and hasattr(mesh, "device")
    
    if use_torch:
        import torch
        M_diag = M.to_dense().diagonal()
        M_inv_sqrt = torch.sparse_coo_tensor(
            torch.stack([torch.arange(len(M_diag), device=mesh.device)] * 2),
            1.0 / torch.sqrt(torch.clamp(M_diag, min=1e-12)),
            (len(M_diag), len(M_diag))
        ).coalesce()
        
        L_sym = M_inv_sqrt @ L @ M_inv_sqrt
        
        # Power iteration to find max eigenvalue
        v = torch.randn(mesh.n_vertices, 1, dtype=L_sym.dtype, device=mesh.device)
        v = v / torch.norm(v)
        for _ in range(20):
            v = L_sym @ v
            v = v / torch.norm(v)
        lambda_max = torch.sum(v * (L_sym @ v))
        
        I = torch.sparse_coo_tensor(
            torch.stack([torch.arange(mesh.n_vertices, device=mesh.device)] * 2),
            torch.ones(mesh.n_vertices, device=mesh.device),
            (mesh.n_vertices, mesh.n_vertices)
        ).coalesce()
        
        L_scaled = (2.0 / lambda_max) * L_sym - I
        return L_scaled
    else:
        from scipy.sparse import diags, identity
        
        M_diag = M.diagonal()
        M_inv_sqrt = diags(1.0 / np.sqrt(np.maximum(M_diag, 1e-12)))
        
        L_sym = M_inv_sqrt @ L @ M_inv_sqrt
        
        # Power iteration to find max eigenvalue
        v = np.random.randn(mesh.n_vertices)
        v = v / np.linalg.norm(v)
        for _ in range(20):
            v = L_sym @ v
            v = v / np.linalg.norm(v)
        lambda_max = np.dot(v, L_sym @ v)
        
        I = identity(mesh.n_vertices, format='csr')
        L_scaled = (2.0 / lambda_max) * L_sym - I
        return L_scaled


def chebyshev_filter(mesh, x: Union[np.ndarray, 'torch.Tensor'], order: int) -> List[Union[np.ndarray, 'torch.Tensor']]:
    """Apply Chebyshev spectral basis filters up to `order` to signal `x`.
    
    Returns:
        A list of filtered signals [T_0(L)x, T_1(L)x, ..., T_k(L)x].
    """
    L_scaled = scaled_laplacian(mesh)
    
    T = [x]
    if order == 0:
        return T
        
    T.append(L_scaled @ x)
    
    for k in range(2, order + 1):
        Tx = 2.0 * (L_scaled @ T[k-1]) - T[k-2]
        T.append(Tx)
        
    return T

__all__ = ["scaled_laplacian", "chebyshev_filter"]