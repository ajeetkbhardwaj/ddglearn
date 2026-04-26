"""Geodesic distance via heat method (Crane et al. 2013).

The heat method computes shortest paths on surfaces by:
1. Solve heat equation for short time: (I - tΔ)u = δ_source
2. Compute gradient: ∇u
3. Integrate: geodesic distance via distance = -∫ ∇u / |∇u|

This is much faster than exact geodesics and numerically stable.
"""
import numpy as np
from typing import Optional, Union, List
import heapq

from operators.gradient import gradient
from operators.divergence import divergence
from pde.poisson import solve_poisson
from operators.exterior_derivative import d0
from operators.hodge_star import hodge_star_0, hodge_star_1

try:
    from scipy.sparse import isspmatrix, csr_matrix
    from scipy.sparse.linalg import spsolve
    _HAS_SCIPY = True
except Exception:
    isspmatrix = lambda x: False
    _HAS_SCIPY = False


def solve_heat_diffusion(
    mesh,
    source_indices: Union[int, List[int], np.ndarray],
    time_scale: float = 1e-3,
    dirichlet_indices: Optional[np.ndarray] = None,
    dirichlet_values: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Solve heat equation (I - t∇²)u = δ for heat values.
    
    source_indices: vertex index or list of indices where delta source is placed
    time_scale: diffusion time (affects geodesic quality)
    dirichlet_indices: optional vertex indices to pin
    dirichlet_values: optional values to pin at dirichlet_indices
    """
    if isinstance(source_indices, int):
        source_indices = [source_indices]

    n = mesh.n_vertices
    rhs = np.zeros(n, dtype=float)
    for src in source_indices:
        rhs[src] = 1.0

    D0 = d0(mesh)
    H1 = hodge_star_1(mesh)
    H0 = hodge_star_0(mesh)

    if _HAS_SCIPY and isspmatrix(D0):
        D0 = csr_matrix(D0)
        H1 = csr_matrix(H1) if not isspmatrix(H1) else H1
        H0 = csr_matrix(H0) if not isspmatrix(H0) else H0
        W = D0.T @ H1 @ D0
        A = (H0 - time_scale * W).tocsr()
        
        if dirichlet_indices is not None and dirichlet_values is not None:
            A = A.tolil()
            for idx, val in zip(dirichlet_indices, dirichlet_values):
                rhs -= A[:, idx].toarray().flatten() * val
                A[idx, :] = 0
                A[:, idx] = 0
                A[idx, idx] = 1.0
                rhs[idx] = val
            A = A.tocsr()
            
        u = spsolve(A, rhs)
        return np.asarray(u).reshape(-1)
    else:
        D0 = D0.toarray() if hasattr(D0, "toarray") else np.asarray(D0)
        H1 = H1.toarray() if hasattr(H1, "toarray") else np.asarray(H1)
        H0 = H0.toarray() if hasattr(H0, "toarray") else np.asarray(H0)
        W = D0.T @ H1 @ D0
        A = H0 - time_scale * W
        
        if dirichlet_indices is not None and dirichlet_values is not None:
            A = A.copy()
            for idx, val in zip(dirichlet_indices, dirichlet_values):
                rhs -= A[:, idx] * val
                A[idx, :] = 0
                A[:, idx] = 0
                A[idx, idx] = 1.0
                rhs[idx] = val
                
        u = np.linalg.solve(A, rhs)
        return u.reshape(-1)


def geodesic_distance(mesh, source_indices: Union[int, List[int], np.ndarray], time_scale: float = 1e-3, method: str = "poisson") -> np.ndarray:
    """Compute geodesic distance from a source vertex (or set of vertices) to all other vertices.

    Uses the heat method: solve short-time heat diffusion, then
    integrate gradient flow. Optionally supports varadhan's formula or graph fast marching.

    Args:
        mesh: The halfedge mesh.
        source_indices: A single vertex index or list of indices.
        time_scale: Time scale parameter for heat diffusion.
        method: "poisson" (standard heat method), "varadhan", or "fmm_graph".

    Returns: distances of shape (n_vertices,)
    """
    if isinstance(source_indices, int):
        sources = [source_indices]
    else:
        sources = list(source_indices)

    if method == "fmm_graph":
        distances = np.full(mesh.n_vertices, np.inf)
        visited = np.zeros(mesh.n_vertices, dtype=bool)
        queue = []
        for s in sources:
            distances[s] = 0.0
            heapq.heappush(queue, (0.0, s))
            
        V = mesh.vertices
        while queue:
            d, u = heapq.heappop(queue)
            if visited[u]:
                continue
            visited[u] = True
            
            for v in mesh.vertex_neighbors(u):
                if visited[v]:
                    continue
                weight = np.linalg.norm(V[u] - V[v])
                if d + weight < distances[v]:
                    distances[v] = d + weight
                    heapq.heappush(queue, (distances[v], v))
        return distances

    # step 1: heat diffusion
    u = solve_heat_diffusion(mesh, sources, time_scale=time_scale)

    if method == "varadhan":
        u_clipped = np.maximum(u, 1e-100)
        distances = np.sqrt(np.maximum(-4.0 * time_scale * np.log(u_clipped), 0.0))
        distances -= np.min(distances[sources])
        for s in sources:
            distances[s] = 0.0
        return distances
        
    # step 2: compute gradient
    grad_u = gradient(mesh, u)

    # step 3: normalize and integrate
    norms = np.linalg.norm(grad_u, axis=-1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    X = -grad_u / norms

    div_X = divergence(mesh, X)

    # Solve Poisson equation: Δ dist = div_X
    distances = solve_poisson(mesh, div_X).reshape(-1)

    # Shift distance to be 0 at the source vertex
    src_mean = np.mean(distances[sources])
    distances -= src_mean

    # Depending on Laplacian sign convention, distances may be negative.
    if np.mean(distances) < 0:
        distances = -distances
        src_mean = np.mean(distances[sources])
        distances -= src_mean

    # Enforce exact 0 at source vertices to clean up numerical Poisson drift
    for s in sources:
        distances[s] = 0.0

    return distances


def heat_method_geodesics(mesh, source_indices: list = None, time_scale: float = 1e-3, batch: bool = False, method: str = "poisson") -> np.ndarray:
    """Batch compute geodesic distances from multiple sources.

    source_indices: list of vertex indices to use as sources (default: single vertex 0)
    batch: If True, computes a single distance field to the nearest source.
           If False, returns array of shape (n_sources, n_vertices) with individual distances.
    method: "poisson" (Heat Method), "varadhan", or "fmm_graph"
    Returns: array of shape (n_sources, n_vertices) or (n_vertices,)
    """
    if source_indices is None:
        source_indices = [0]

    if batch:
        return geodesic_distance(mesh, source_indices, time_scale=time_scale, method=method)

    n_sources = len(source_indices)
    n_v = mesh.n_vertices
    distances = np.zeros((n_sources, n_v), dtype=float)

    for i, src in enumerate(source_indices):
        distances[i, :] = geodesic_distance(mesh, src, time_scale=time_scale, method=method)

    return distances


__all__ = ["geodesic_distance", "heat_method_geodesics", "solve_heat_diffusion"]
