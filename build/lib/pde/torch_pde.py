"""GPU-accelerated PDE solvers using PyTorch.

This module provides:
- Poisson solver (GPU-accelerated)
- Heat equation solver (implicit Euler)
- Wave equation solver
- Geodesic distance via heat method
- Conjugate gradient solver
"""

import numpy as np
from typing import Optional, Union, Tuple

try:
    import torch
    from torch import zeros

    _HAS_TORCH = True
except ImportError:
    torch = None
    _HAS_TORCH = False

from core.torch_mesh import TorchHalfEdgeMesh
from core.performance import ensure_tensor
from operators.torch_operators import d0, hodge_star_0, hodge_star_1


# =============================================================================
# Linear Solver Utilities
# =============================================================================


def cg_solve(
    A: torch.Tensor,
    b: torch.Tensor,
    x0: Optional[torch.Tensor] = None,
    max_iter: int = 1000,
    tol: float = 1e-8,
    preconditioner: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Conjugate Gradient solver for sparse systems.

    Args:
        A: Sparse system matrix
        b: Right-hand side
        x0: Initial guess
        max_iter: Maximum iterations
        tol: Convergence tolerance
        preconditioner: Optional preconditioner (diagonal)

    Returns:
        Solution vector x
    """
    if not _HAS_TORCH:
        return _cg_numpy(A, b, x0, max_iter, tol)

    device = b.device
    n = b.shape[0]

    if x0 is None:
        x = zeros(n, device=device)
    else:
        x = x0.clone()

    r = b - A @ x

    if preconditioner is not None:
        z = r / preconditioner
    else:
        z = r.clone()

    p = z.clone()
    rz_old = torch.dot(r, z)

    for i in range(max_iter):
        Ap = A @ p

        alpha = rz_old / torch.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap

        if preconditioner is not None:
            z = r / preconditioner
        else:
            z = r.clone()

        rz_new = torch.dot(r, z)

        if torch.sqrt(rz_new) < tol:
            break

        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new

    return x


def _cg_numpy(A, b, x0, max_iter, tol):
    """NumPy fallback for CG solver."""
    from scipy.sparse.linalg import cg

    return cg(A, b, x0=x0, maxiter=max_iter, tol=tol)[0]


# =============================================================================
# Poisson Solver
# =============================================================================


def solve_poisson(
    mesh: TorchHalfEdgeMesh,
    f: Union[np.ndarray, torch.Tensor],
    pin_index: Optional[int] = 0,
    pin_value: float = 0.0,
    use_cg: bool = True,
    dirichlet_indices: Optional[torch.Tensor] = None,
    dirichlet_values: Optional[torch.Tensor] = None,
    neumann_indices: Optional[torch.Tensor] = None,
    neumann_values: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Solve Poisson equation: Δu = f on the mesh.

    Args:
        mesh: The mesh
        f: Right-hand side (vertex values)
        pin_index: Vertex to pin (for well-posedness)
        pin_value: Value at pinned vertex
        use_cg: Use CG solver (faster for large systems)

    Returns:
        Solution u at vertices
    """
    if not _HAS_TORCH:
        return _solve_poisson_numpy(mesh, f, pin_index, pin_value, dirichlet_indices, dirichlet_values, neumann_indices, neumann_values)

    device = mesh.device
    f_t = ensure_tensor(f, device=device)

    D0 = d0(mesh)
    H1 = hodge_star_1(mesh)
    H0 = hodge_star_0(mesh)

    W = D0.T @ H1 @ D0
    rhs = H0 @ f_t

    n = mesh.n_vertices

    if pin_index is not None:
        W = W.to_dense()
        W[pin_index, :] = 0
        W[:, pin_index] = 0
        W[pin_index, pin_index] = 1.0
        rhs[pin_index] = pin_value

        W = W.to_sparse()

    if use_cg:
        diag = H0.to_dense().diagonal()
        diag = torch.clamp(diag, min=1e-12)
        precond = diag

        return cg_solve(W, rhs, max_iter=n, tol=1e-8, preconditioner=precond)
    else:
        W_dense = W.to_dense()
        return torch.linalg.solve(W_dense, rhs)


def _solve_poisson_numpy(mesh, f, pin_index, pin_value, dirichlet_indices=None, dirichlet_values=None, neumann_indices=None, neumann_values=None):
    """NumPy fallback for Poisson solver."""
    from pde.poisson import solve_poisson as _sp
    
    d_idx = dirichlet_indices.cpu().numpy() if hasattr(dirichlet_indices, "cpu") else dirichlet_indices
    d_val = dirichlet_values.cpu().numpy() if hasattr(dirichlet_values, "cpu") else dirichlet_values
    n_idx = neumann_indices.cpu().numpy() if hasattr(neumann_indices, "cpu") else neumann_indices
    n_val = neumann_values.cpu().numpy() if hasattr(neumann_values, "cpu") else neumann_values
    
    res = _sp(
        mesh, f, pin_index=pin_index, pin_value=pin_value,
        dirichlet_indices=d_idx, dirichlet_values=d_val,
        neumann_indices=n_idx, neumann_values=n_val
    )
    if _HAS_TORCH:
        return torch.tensor(res, device=f.device if hasattr(f, 'device') else 'cpu')
    return res


# =============================================================================
# Heat Equation
# =============================================================================


def implicit_heat_step(
    mesh: TorchHalfEdgeMesh,
    u: Union[np.ndarray, torch.Tensor],
    t: float,
    pin_index: Optional[int] = 0,
    dirichlet_indices: Optional[torch.Tensor] = None,
    dirichlet_values: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Single implicit Euler heat step.

    Solves: (H0 - t * W) u_next = H0 * u

    Args:
        mesh: The mesh
        u: Current state
        t: Time step
        pin_index: Pinned vertex index

    Returns:
        Next state u_next
    """
    if not _HAS_TORCH:
        return _implicit_heat_step_numpy(mesh, u, t, pin_index, dirichlet_indices, dirichlet_values)

    device = mesh.device
    u_t = ensure_tensor(u, device=device)

    D0 = d0(mesh)
    H1 = hodge_star_1(mesh)
    H0 = hodge_star_0(mesh)

    W = D0.T @ H1 @ D0
    A = H0 - t * W
    rhs = H0 @ u_t

    if dirichlet_indices is not None and dirichlet_values is not None:
        d_idx = ensure_tensor(dirichlet_indices, device=device).long()
        d_val = ensure_tensor(dirichlet_values, device=device)
        A = A.to_dense()
        for idx, val in zip(d_idx, d_val):
            rhs -= A[:, idx] * val
            A[idx, :] = 0
            A[:, idx] = 0
            A[idx, idx] = 1.0
            rhs[idx] = val
        A = A.to_sparse()
    elif pin_index is not None:
        A = A.to_dense()
        A[pin_index, :] = 0
        A[:, pin_index] = 0
        A[pin_index, pin_index] = 1.0
        rhs[pin_index] = 0.0
        A = A.to_sparse()

    return torch.linalg.solve(A.to_dense(), rhs)


def _implicit_heat_step_numpy(mesh, u, t, pin_index, dirichlet_indices=None, dirichlet_values=None):
    """NumPy fallback for heat step."""
    from pde.heat import implicit_heat_step as _ihs
    d_idx = dirichlet_indices.cpu().numpy() if hasattr(dirichlet_indices, "cpu") else dirichlet_indices
    d_val = dirichlet_values.cpu().numpy() if hasattr(dirichlet_values, "cpu") else dirichlet_values
    res = _ihs(mesh, u, t, pin_index=pin_index, dirichlet_indices=d_idx, dirichlet_values=d_val)
    if _HAS_TORCH:
        return torch.tensor(res, device=u.device if hasattr(u, 'device') else 'cpu')
    return res


def implicit_heat(
    mesh: TorchHalfEdgeMesh,
    u0: Union[np.ndarray, torch.Tensor],
    t: float,
    steps: int = 1,
    pin_index: Optional[int] = 0,
    dirichlet_indices: Optional[torch.Tensor] = None,
    dirichlet_values: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run multiple implicit heat steps."""
    u = ensure_tensor(u0, device=mesh.device)

    for _ in range(steps):
        u = implicit_heat_step(mesh, u, t, pin_index, dirichlet_indices, dirichlet_values)

    return u


# =============================================================================
# Wave Equation
# =============================================================================


def wave_step(
    mesh: TorchHalfEdgeMesh,
    u_prev: Union[np.ndarray, torch.Tensor],
    u_curr: Union[np.ndarray, torch.Tensor],
    dt: float,
    pin_index: Optional[int] = 0,
) -> torch.Tensor:
    """Single wave equation time step (central differences).

    A * u^{n+1} = 2*A*u^n - A*u^{n-1} - dt^2 * W * u^n
    """
    if not _HAS_TORCH:
        return _wave_step_numpy(mesh, u_prev, u_curr, dt, pin_index)

    device = mesh.device
    u_prev_t = ensure_tensor(u_prev, device=device)
    u_curr_t = ensure_tensor(u_curr, device=device)

    D0 = d0(mesh)
    H1 = hodge_star_1(mesh)
    H0 = hodge_star_0(mesh)

    W = D0.T @ H1 @ D0

    rhs = 2.0 * (H0 @ u_curr_t) - (H0 @ u_prev_t) - (dt * dt) * (W @ u_curr_t)

    A = H0.to_dense()

    if pin_index is not None:
        A[pin_index, :] = 0
        A[:, pin_index] = 0
        A[pin_index, pin_index] = 1.0
        rhs[pin_index] = 0.0

    return torch.linalg.solve(A, rhs)


def _wave_step_numpy(mesh, u_prev, u_curr, dt, pin_index):
    """NumPy fallback for wave step."""
    from scipy.sparse.linalg import spsolve
    from operators.exterior_derivative import d0
    from operators.hodge_star import hodge_star_0, hodge_star_1

    D0 = d0(mesh)
    H1 = hodge_star_1(mesh)
    H0 = hodge_star_0(mesh)

    W = D0.T.dot(H1.dot(D0))
    rhs = 2.0 * (H0.dot(u_curr)) - (H0.dot(u_prev)) - (dt * dt) * (W.dot(u_curr))

    A = H0.toarray()
    if pin_index is not None:
        A[pin_index, :] = 0
        A[:, pin_index] = 0
        A[pin_index, pin_index] = 1.0
        rhs[pin_index] = 0.0

    return np.linalg.solve(A, rhs)


def simulate_wave(
    mesh: TorchHalfEdgeMesh,
    u0: Union[np.ndarray, torch.Tensor],
    dt: float,
    steps: int,
    u1: Optional[Union[np.ndarray, torch.Tensor]] = None,
    pin_index: Optional[int] = 0,
) -> torch.Tensor:
    """Simulate wave equation."""
    if not _HAS_TORCH:
        return _simulate_wave_numpy(mesh, u0, dt, steps, u1, pin_index)

    device = mesh.device
    u_prev = ensure_tensor(u0, device=device)
    u_curr = (
        u1
        if u1 is not None
        else u0.clone()
        if hasattr(u0, "clone")
        else ensure_tensor(u0, device=device)
    )

    for _ in range(steps):
        u_next = wave_step(mesh, u_prev, u_curr, dt, pin_index)
        u_prev, u_curr = u_curr, u_next

    return u_curr


def _simulate_wave_numpy(mesh, u0, dt, steps, u1, pin_index):
    """NumPy fallback for wave simulation."""
    from pde.wave import simulate_wave as _sim

    return _sim(mesh, u0, dt, steps, u1, pin_index)


# =============================================================================
# Geodesic Distance (Heat Method)
# =============================================================================


def geodesic_distance(
    mesh: TorchHalfEdgeMesh, 
    source_indices: Union[int, list, np.ndarray, torch.Tensor], 
    time_scale: float = 1e-3,
    method: str = "poisson",
) -> torch.Tensor:
    """Compute geodesic distance using heat method.

    Args:
        mesh: The mesh
        source_indices: Source vertex index or list of indices
        time_scale: Diffusion time (affects quality)
        method: "poisson" (standard heat method), "varadhan", or "fmm_graph"

    Returns:
        Distance from source to all vertices
    """
    if not _HAS_TORCH:
        return _geodesic_distance_numpy(mesh, source_indices, time_scale, method)

    device = mesh.device
    n = mesh.n_vertices

    if isinstance(source_indices, int):
        sources = [source_indices]
    elif isinstance(source_indices, torch.Tensor):
        sources = source_indices.tolist()
    else:
        sources = list(source_indices)

    if method == "fmm_graph":
        import heapq
        distances = torch.full((n,), float('inf'), device=device)
        visited = torch.zeros(n, dtype=torch.bool, device=device)
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
                weight = torch.norm(V[u] - V[v]).item()
                if d + weight < distances[v].item():
                    distances[v] = d + weight
                    heapq.heappush(queue, (distances[v].item(), v))
        return distances

    rhs = torch.zeros(n, device=device)
    for s in sources:
        rhs[s] = 1.0

    D0 = d0(mesh)
    H1 = hodge_star_1(mesh)
    H0 = hodge_star_0(mesh)

    W = D0.T @ H1 @ D0
    A = H0 - time_scale * W

    u = torch.linalg.solve(A.to_dense(), rhs)

    if method == "varadhan":
        u_clipped = torch.clamp(u, min=1e-100)
        distances = torch.sqrt(torch.clamp(-4.0 * time_scale * torch.log(u_clipped), min=0.0))
        src_mean = torch.mean(distances[sources])
        distances -= src_mean
        for s in sources:
            distances[s] = 0.0
        return distances

    grad_u = D0 @ u
    norms = torch.abs(grad_u)
    norms = torch.clamp(norms, min=1e-12)
    X = -grad_u / norms

    div_X = D0.T @ (H1 @ X)
    distances = solve_poisson(mesh, div_X, use_cg=False)

    src_mean = torch.mean(distances[sources])
    distances -= src_mean

    if torch.mean(distances) < 0:
        distances = -distances
        src_mean = torch.mean(distances[sources])
        distances -= src_mean

    for s in sources:
        distances[s] = 0.0

    return distances


def _geodesic_distance_numpy(mesh, source_indices, time_scale, method="poisson"):
    """NumPy fallback for geodesic distance."""
    from pde.geodesics import geodesic_distance as _gd

    res = _gd(mesh, source_indices, time_scale, method=method)
    if _HAS_TORCH:
        return torch.tensor(res, device=mesh.device if hasattr(mesh, 'device') else 'cpu')
    return res


def heat_method_geodesics(
    mesh: TorchHalfEdgeMesh, 
    source_indices: Optional[list] = None, 
    time_scale: float = 1e-3, 
    batch: bool = False, 
    method: str = "poisson"
) -> torch.Tensor:
    """Compute geodesic distances from multiple sources."""
    if source_indices is None:
        source_indices = [0]

    if batch:
        return geodesic_distance(mesh, source_indices, time_scale, method)

    distances = []
    for src in source_indices:
        distances.append(geodesic_distance(mesh, src, time_scale, method))

    return torch.stack(distances)


# =============================================================================
# Hodge Decomposition
# =============================================================================


def hodge_decomposition(
    mesh: TorchHalfEdgeMesh, v_edges: Union[np.ndarray, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decompose 1-form into curl-free + div-free + harmonic parts."""
    if not _HAS_TORCH:
        return _hodge_decomposition_numpy(mesh, v_edges)

    from operators.torch_operators import gradient, divergence, curl_vector

    device = mesh.device
    v = ensure_tensor(v_edges, device=device)

    curl_v = curl_vector(mesh, v)
    n_v = mesh.n_vertices
    n_f = mesh.n_faces

    curl_at_vertex = torch.zeros(n_v, device=device)
    face_to_vertex_count = torch.zeros(n_v, device=device)

    f_np = mesh.faces.cpu().numpy()
    for f_idx, f in enumerate(f_np):
        for vi in f:
            curl_at_vertex[vi] += curl_v[f_idx]
            face_to_vertex_count[vi] += 1

    curl_at_vertex = curl_at_vertex / torch.clamp(face_to_vertex_count, min=1)

    g = solve_poisson(mesh, curl_at_vertex, pin_index=0, pin_value=0.0)
    grad_g = gradient(mesh, g)

    div_v = divergence(mesh, v)
    f = solve_poisson(mesh, div_v, pin_index=0, pin_value=0.0)
    grad_f = gradient(mesh, f)

    harmonic_part = v - grad_f - grad_g

    return grad_f, grad_g, harmonic_part


def _hodge_decomposition_numpy(mesh, v_edges):
    """NumPy fallback for Hodge decomposition."""
    from pde.hodge_decomposition import hodge_decomposition as _hd

    return _hd(mesh, v_edges)


__all__ = [
    "solve_poisson",
    "implicit_heat_step",
    "implicit_heat",
    "wave_step",
    "simulate_wave",
    "geodesic_distance",
    "heat_method_geodesics",
    "hodge_decomposition",
    "cg_solve",
]
