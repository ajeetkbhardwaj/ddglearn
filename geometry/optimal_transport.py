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
        from operators.torch_operators import d0, hodge_star_0, hodge_star_1
        
        D0 = d0(mesh)
        H1 = hodge_star_1(mesh)
        H0 = hodge_star_0(mesh)
        W = D0.T @ H1 @ D0
        
        A = (H0 - t * W).to_dense()
        LU, pivots = torch.linalg.lu_factor(A)
        H0_dense = H0.to_dense()
        
        u = torch.ones_like(p)
        v = torch.ones_like(q)
        
        for _ in range(max_iter):
            Hv = torch.linalg.lu_solve(LU, pivots, (H0_dense @ v).unsqueeze(1)).squeeze(1)
            u_new = p / torch.clamp(Hv, min=1e-12)
            
            Hu = torch.linalg.lu_solve(LU, pivots, (H0_dense @ u_new).unsqueeze(1)).squeeze(1)
            v_new = q / torch.clamp(Hu, min=1e-12)
            
            err = torch.max(torch.abs(u_new - u)).item()
            u, v = u_new, v_new
            
            if err < tol:
                break
                
        # Approximate Wasserstein distance via dual objective
        dist = t * torch.sum(p * torch.log(torch.clamp(u, min=1e-12)) + q * torch.log(torch.clamp(v, min=1e-12)))
        return dist, u, v
        
    else:
        from operators.exterior_derivative import d0
        from operators.hodge_star import hodge_star_0, hodge_star_1
        
        try:
            from scipy.sparse import csr_matrix, isspmatrix
            from scipy.sparse.linalg import factorized
            _LOCAL_HAS_SCIPY = True
        except Exception:
            _LOCAL_HAS_SCIPY = False

        D0 = d0(mesh)
        H0 = hodge_star_0(mesh)
        H1 = hodge_star_1(mesh)
        
        u = np.ones_like(p)
        v = np.ones_like(q)
        
        if _LOCAL_HAS_SCIPY:
            D0 = csr_matrix(D0) if not isspmatrix(D0) else D0
            H1 = csr_matrix(H1) if not isspmatrix(H1) else H1
            H0 = csr_matrix(H0) if not isspmatrix(H0) else H0
            
            W = D0.T @ H1 @ D0
            A = (H0 - t * W).tocsc()
            solve_A = factorized(A)
            
            for _ in range(max_iter):
                Hv = solve_A(H0 @ v)
                u_new = p / np.maximum(Hv, 1e-12)
                
                Hu = solve_A(H0 @ u_new)
                v_new = q / np.maximum(Hu, 1e-12)
                
                err = np.max(np.abs(u_new - u))
                u, v = u_new, v_new
                
                if err < tol:
                    break
        else:
            D0_dense = D0.toarray() if hasattr(D0, "toarray") else np.asarray(D0)
            H1_dense = H1.toarray() if hasattr(H1, "toarray") else np.asarray(H1)
            H0_dense = H0.toarray() if hasattr(H0, "toarray") else np.asarray(H0)
            W = D0_dense.T @ H1_dense @ D0_dense
            A = H0_dense - t * W
            
            for _ in range(max_iter):
                Hv = np.linalg.solve(A, H0_dense @ v)
                u_new = p / np.maximum(Hv, 1e-12)
                
                Hu = np.linalg.solve(A, H0_dense @ u_new)
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
        from operators.torch_operators import d0, hodge_star_0, hodge_star_1
        
        if weights is None:
            weights = torch.ones(K, device=distributions.device) / K
            
        D0 = d0(mesh)
        H1 = hodge_star_1(mesh)
        H0 = hodge_star_0(mesh)
        W = D0.T @ H1 @ D0
        A = (H0 - t * W).to_dense()
        LU, pivots = torch.linalg.lu_factor(A)
        H0_dense = H0.to_dense()
            
        v = torch.ones_like(distributions)
        u = torch.ones_like(distributions)
        b = torch.ones(distributions.shape[1], device=distributions.device) / distributions.shape[1]
        
        for _ in range(max_iter):
            hv = torch.stack([torch.linalg.lu_solve(LU, pivots, (H0_dense @ v[k]).unsqueeze(1)).squeeze(1) for k in range(K)])
            u = distributions / torch.clamp(hv, min=1e-12)
            
            hu = torch.stack([torch.linalg.lu_solve(LU, pivots, (H0_dense @ u[k]).unsqueeze(1)).squeeze(1) for k in range(K)])
            
            b_new = torch.exp(torch.sum(weights.unsqueeze(1) * torch.log(torch.clamp(u * hu, min=1e-12)), dim=0))
            b_new = b_new / torch.sum(b_new)
            
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