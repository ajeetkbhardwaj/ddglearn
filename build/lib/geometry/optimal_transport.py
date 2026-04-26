"""Optimal Transport Maps on Surfaces.

Implements Convolutional Wasserstein Distances (Solomon et al. 2015),
which uses the Heat Equation to approximate the Sinkhorn kernel convolution.
This computes optimal transport in O(N) time without an O(N^2) distance matrix.
"""
import numpy as np

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def sinkhorn_wasserstein(mesh, p, q, t: float = 1e-3, max_iter: int = 150, tol: float = 1e-5):
    """Compute the Optimal Transport (Wasserstein) distance between distributions.

    Args:
        mesh: The HalfEdgeMesh or TorchHalfEdgeMesh.
        p: Source mass distribution array (shape: n_vertices).
        q: Target mass distribution array (shape: n_vertices).
        t: Diffusion time (controls entropy regularization / blurring).
        max_iter: Maximum Sinkhorn iterations.
        tol: Convergence tolerance.

    Returns:
        dist: Approximate Wasserstein distance.
        u: Left dual scaling variable.
        v: Right dual scaling variable.
    """
    is_torch = _HAS_TORCH and hasattr(p, "device") and isinstance(p, torch.Tensor)
    
    if is_torch:
        from pde.torch_pde import implicit_heat
        u = torch.ones_like(p)
        v = torch.ones_like(q)
        
        for _ in range(max_iter):
            # Heat diffusion is symmetric: H(v) acts as K * v
            Hv = implicit_heat(mesh, v, t, steps=1, pin_index=None)
            u_new = p / torch.clamp(Hv, min=1e-12)
            
            Hu = implicit_heat(mesh, u_new, t, steps=1, pin_index=None)
            v_new = q / torch.clamp(Hu, min=1e-12)
            
            err = torch.max(torch.abs(u_new - u)).item()
            u, v = u_new, v_new
            
            if err < tol:
                break
                
        # Approximate Wasserstein distance via dual objective
        dist = t * torch.sum(p * torch.log(torch.clamp(u, min=1e-12)) + q * torch.log(torch.clamp(v, min=1e-12)))
        return dist, u, v
        
    else:
        from pde.heat import implicit_heat
        u = np.ones_like(p)
        v = np.ones_like(q)
        
        for _ in range(max_iter):
            Hv = implicit_heat(mesh, v, t, steps=1, pin_index=None)
            u_new = p / np.maximum(Hv, 1e-12)
            
            Hu = implicit_heat(mesh, u_new, t, steps=1, pin_index=None)
            v_new = q / np.maximum(Hu, 1e-12)
            
            err = np.max(np.abs(u_new - u))
            u, v = u_new, v_new
            
            if err < tol:
                break
                
        dist = t * np.sum(p * np.log(np.maximum(u, 1e-12)) + q * np.log(np.maximum(v, 1e-12)))
        return dist, u, v


def wasserstein_barycenter(mesh, distributions, weights=None, t: float = 1e-3, max_iter: int = 150, tol: float = 1e-5):
    """Compute the Wasserstein barycenter (geometric mean) of multiple distributions.
    
    Args:
        mesh: The HalfEdgeMesh or TorchHalfEdgeMesh.
        distributions: (K, n_vertices) array of K distinct probability distributions.
        weights: (K,) array of weights for the barycenter (default: uniform).
        t: Diffusion time.
        
    Returns:
        b: (n_vertices,) The computed barycenter distribution.
    """
    is_torch = _HAS_TORCH and hasattr(distributions, "device") and isinstance(distributions, torch.Tensor)
    K = distributions.shape[0]
    
    if is_torch:
        from pde.torch_pde import implicit_heat
        if weights is None:
            weights = torch.ones(K, device=distributions.device) / K
            
        v = torch.ones_like(distributions)
        u = torch.ones_like(distributions)
        b = torch.ones(distributions.shape[1], device=distributions.device) / distributions.shape[1]
        
        for _ in range(max_iter):
            # Smooth v
            hv = torch.stack([implicit_heat(mesh, v[k], t, steps=1, pin_index=None) for k in range(K)])
            u = distributions / torch.clamp(hv, min=1e-12)
            
            # Smooth u
            hu = torch.stack([implicit_heat(mesh, u[k], t, steps=1, pin_index=None) for k in range(K)])
            
            # Update barycenter
            b_new = torch.exp(torch.sum(weights.unsqueeze(1) * torch.log(torch.clamp(u * hu, min=1e-12)), dim=0))
            b_new = b_new / torch.sum(b_new)
            
            # Update v
            v = b_new.unsqueeze(0) / torch.clamp(hu, min=1e-12)
            
            err = torch.max(torch.abs(b_new - b)).item()
            b = b_new
            
            if err < tol:
                break
                
        return b
        
    else:
        # NumPy fallback logic
        raise NotImplementedError("Wasserstein Barycenters currently requires the PyTorch backend for batch parallelization.")


__all__ = ["sinkhorn_wasserstein", "wasserstein_barycenter"]