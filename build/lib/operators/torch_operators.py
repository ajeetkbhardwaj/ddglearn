"""PyTorch-based DEC operators with GPU acceleration.

This module provides GPU-accelerated implementations of:
- Exterior derivatives (d0, d1)
- Hodge stars (*0, *1, *2)
- Laplacian operator
- Gradient, divergence, curl
"""

import numpy as np
import threading
from typing import Optional, Tuple, Union, List

try:
    import torch
    from torch import sparse_coo_tensor

    _HAS_TORCH = True
except ImportError:
    torch = None
    _HAS_TORCH = False

from core.performance import ensure_tensor
from core.torch_mesh import TorchHalfEdgeMesh


# =============================================================================
# Utility Functions
# =============================================================================


def _edge_list_and_map_torch(mesh: TorchHalfEdgeMesh) -> Tuple[np.ndarray, dict]:
    """Get edge list and map for the mesh."""
    edges, edge_map = mesh.edge_list, mesh.edge_map
    return edges, edge_map


# =============================================================================
# Exterior Derivatives
# =============================================================================


class ExteriorDerivativeCache:
    """Thread-safe cache for exterior derivative matrices."""

    def __init__(self):
        self._lock = threading.Lock()
        self._d0 = {}
        self._d1 = {}

    def get_d0(self, mesh: TorchHalfEdgeMesh):
        key = id(mesh)
        with self._lock:
            if key not in self._d0:
                self._d0[key] = self._compute_d0(mesh)
            return self._d0[key]

    def get_d1(self, mesh: TorchHalfEdgeMesh):
        key = id(mesh)
        with self._lock:
            if key not in self._d1:
                self._d1[key] = self._compute_d1(mesh)
            return self._d1[key]

    def _compute_d0(self, mesh: TorchHalfEdgeMesh) -> torch.Tensor:
        """Compute d0: vertices -> edges."""
        edges, edge_map = _edge_list_and_map_torch(mesh)
        n_e = len(edges)
        n_v = mesh.n_vertices

        rows = []
        cols = []
        data = []

        for ei, (u, v) in enumerate(edges):
            rows.extend([ei, ei])
            cols.extend([u, v])
            data.extend([-1.0, 1.0])

        if _HAS_TORCH:
            indices = torch.tensor([rows, cols], dtype=torch.long, device=mesh.device)
            values = torch.tensor(data, dtype=torch.float64, device=mesh.device)
            return sparse_coo_tensor(indices, values, (n_e, n_v), device=mesh.device)
        else:
            from scipy.sparse import coo_matrix

            return coo_matrix((data, (rows, cols)), shape=(n_e, n_v))

    def _compute_d1(self, mesh: TorchHalfEdgeMesh) -> torch.Tensor:
        """Compute d1: edges -> faces."""
        edges, edge_map = _edge_list_and_map_torch(mesh)
        n_e = len(edges)
        n_f = mesh.n_faces

        rows = []
        cols = []
        data = []

        f_np = mesh.faces.cpu().numpy() if _HAS_TORCH else mesh.faces

        for fi, f in enumerate(f_np):
            v0, v1, v2 = int(f[0]), int(f[1]), int(f[2])
            face_edges = [(v0, v1), (v1, v2), (v2, v0)]

            for a, b in face_edges:
                key = (min(a, b), max(a, b))
                ei = edge_map[key]
                stored_a, stored_b = edges[ei]
                sign = 1.0 if (a == stored_a and b == stored_b) else -1.0

                rows.append(fi)
                cols.append(ei)
                data.append(sign)

        if _HAS_TORCH:
            indices = torch.tensor([rows, cols], dtype=torch.long, device=mesh.device)
            values = torch.tensor(data, dtype=torch.float64, device=mesh.device)
            return sparse_coo_tensor(indices, values, (n_f, n_e), device=mesh.device)
        else:
            from scipy.sparse import coo_matrix

            return coo_matrix((data, (rows, cols)), shape=(n_f, n_e))

    def invalidate(self, mesh: TorchHalfEdgeMesh):
        key = id(mesh)
        with self._lock:
            self._d0.pop(key, None)
            self._d1.pop(key, None)


_exterior_cache = ExteriorDerivativeCache()


def d0(mesh: TorchHalfEdgeMesh) -> Union[torch.Tensor, np.ndarray]:
    """Exterior derivative d0: vertices -> edges.

    Returns sparse matrix of shape (n_edges, n_vertices).
    """
    return _exterior_cache.get_d0(mesh)


def d1(mesh: TorchHalfEdgeMesh) -> Union[torch.Tensor, np.ndarray]:
    """Exterior derivative d1: edges -> faces.

    Returns sparse matrix of shape (n_faces, n_edges).
    """
    return _exterior_cache.get_d1(mesh)


# =============================================================================
# Hodge Star Operators
# =============================================================================


class HodgeStarCache:
    """Thread-safe cache for Hodge star matrices."""

    def __init__(self):
        self._lock = threading.Lock()
        self._star0 = {}
        self._star1 = {}
        self._star2 = {}

    def get_star0(self, mesh: TorchHalfEdgeMesh):
        key = id(mesh)
        with self._lock:
            if key not in self._star0:
                self._star0[key] = self._compute_star0(mesh)
            return self._star0[key]

    def get_star1(self, mesh: TorchHalfEdgeMesh):
        key = id(mesh)
        with self._lock:
            if key not in self._star1:
                self._star1[key] = self._compute_star1(mesh)
            return self._star1[key]

    def get_star2(self, mesh: TorchHalfEdgeMesh):
        key = id(mesh)
        with self._lock:
            if key not in self._star2:
                self._star2[key] = self._compute_star2(mesh)
            return self._star2[key]

    def _compute_star0(self, mesh: TorchHalfEdgeMesh):
        """Hodge star *0: vertex areas (diagonal matrix)."""
        vertex_areas = mesh.vertex_area_voronoi_tensor

        if _HAS_TORCH:
            n = mesh.n_vertices
            indices = torch.arange(n, device=mesh.device)
            return sparse_coo_tensor([indices, indices], vertex_areas, (n, n), device=mesh.device)
        else:
            return np.diag(vertex_areas)

    def _compute_star1(self, mesh: TorchHalfEdgeMesh):
        """Hodge star *1: edge dual measures."""
        edges, edge_map = _edge_list_and_map_torch(mesh)
        n_e = len(edges)

        face_areas = mesh.face_areas_tensor

        edge_to_faces = {i: [] for i in range(n_e)}
        f_np = mesh.faces.cpu().numpy() if _HAS_TORCH else mesh.faces

        for fi, f in enumerate(f_np):
            for a, b in [(int(f[0]), int(f[1])), (int(f[1]), int(f[2])), (int(f[2]), int(f[0]))]:
                key = (min(a, b), max(a, b))
                ei = edge_map[key]
                edge_to_faces[ei].append(fi)

        weights = torch.zeros(n_e, dtype=torch.float64, device=mesh.device)
        V = mesh.vertices

        for ei, (a, b) in enumerate(edges):
            pa = V[a]
            pb = V[b]
            L = torch.norm(pb - pa)

            adj = edge_to_faces[ei]
            if len(adj) == 0:
                dual_measure = 1.0
            else:
                dual_measure = face_areas[adj].mean()

            weights[ei] = L / max(dual_measure, 1e-12)

        if _HAS_TORCH:
            indices = torch.arange(n_e, device=mesh.device)
            return sparse_coo_tensor([indices, indices], weights, (n_e, n_e), device=mesh.device)
        else:
            return np.diag(weights)

    def _compute_star2(self, mesh: TorchHalfEdgeMesh):
        """Hodge star *2: face areas (diagonal matrix)."""
        face_areas = mesh.face_areas_tensor
        n_f = mesh.n_faces

        if _HAS_TORCH:
            indices = torch.arange(n_f, device=mesh.device)
            return sparse_coo_tensor([indices, indices], face_areas, (n_f, n_f), device=mesh.device)
        else:
            return np.diag(face_areas)

    def invalidate(self, mesh: TorchHalfEdgeMesh):
        key = id(mesh)
        with self._lock:
            self._star0.pop(key, None)
            self._star1.pop(key, None)
            self._star2.pop(key, None)


_hodge_cache = HodgeStarCache()


def hodge_star_0(mesh: TorchHalfEdgeMesh) -> Union[torch.Tensor, np.ndarray]:
    """Hodge star *0 (vertex areas)."""
    return _hodge_cache.get_star0(mesh)


def hodge_star_1(mesh: TorchHalfEdgeMesh) -> Union[torch.Tensor, np.ndarray]:
    """Hodge star *1 (edge dual measures)."""
    return _hodge_cache.get_star1(mesh)


def hodge_star_2(mesh: TorchHalfEdgeMesh) -> Union[torch.Tensor, np.ndarray]:
    """Hodge star *2 (face areas)."""
    return _hodge_cache.get_star2(mesh)


# =============================================================================
# Laplacian
# =============================================================================


class LaplacianCache:
    """Thread-safe cache for Laplacian matrix."""

    def __init__(self):
        self._lock = threading.Lock()
        self._laplacian = {}

    def get(self, mesh: TorchHalfEdgeMesh):
        key = id(mesh)
        with self._lock:
            if key not in self._laplacian:
                self._laplacian[key] = self._compute(mesh)
            return self._laplacian[key]

    def _compute(self, mesh: TorchHalfEdgeMesh):
        """Compute DEC Laplacian: L = *0^{-1} d0^T *1 d0"""
        D0 = d0(mesh)
        H1 = hodge_star_1(mesh)
        H0 = hodge_star_0(mesh)

        if _HAS_TORCH:
            W = D0.T @ H1 @ D0
            diag = H0.to_dense().diagonal()
            diag = torch.clamp(diag, min=1e-12)
            H0_inv_diag = 1.0 / diag

            n = mesh.n_vertices
            indices = torch.arange(n, device=mesh.device)
            H0_inv = sparse_coo_tensor([indices, indices], H0_inv_diag, (n, n), device=mesh.device)
            return H0_inv @ W
        else:
            W = D0.T.dot(H1.dot(D0))
            diag = np.diag(H0.toarray())
            diag = np.maximum(diag, 1e-12)
            H0_inv = np.diag(1.0 / diag)
            return H0_inv.dot(W)

    def invalidate(self, mesh: TorchHalfEdgeMesh):
        key = id(mesh)
        with self._lock:
            self._laplacian.pop(key, None)


_laplacian_cache = LaplacianCache()


def laplacian_0(mesh: TorchHalfEdgeMesh) -> Union[torch.Tensor, np.ndarray]:
    """DEC Laplacian for 0-forms (vertex scalar Laplacian)."""
    return _laplacian_cache.get(mesh)


# =============================================================================
# Vector Calculus Operators
# =============================================================================


def gradient(
    mesh: TorchHalfEdgeMesh, u: Union[torch.Tensor, np.ndarray]
) -> Union[torch.Tensor, np.ndarray]:
    """Compute gradient of scalar field u.

    Returns edge values: grad(u) = d0^T * u
    """
    D0 = d0(mesh)

    if _HAS_TORCH and isinstance(u, torch.Tensor):
        return D0 @ u
    else:
        if isinstance(u, np.ndarray):
            u = ensure_tensor(u, device=str(mesh.device))
        return (D0 @ u).cpu().numpy()


def divergence(
    mesh: TorchHalfEdgeMesh, v: Union[torch.Tensor, np.ndarray]
) -> Union[torch.Tensor, np.ndarray]:
    """Compute divergence of 1-form v.

    Returns vertex values: div(v) = -d1^T *2^{-1} v
    """
    D0_mat = d0(mesh)
    H1 = hodge_star_1(mesh)
    H0 = hodge_star_0(mesh)

    if _HAS_TORCH and isinstance(v, torch.Tensor):
        w = H1 @ v
        div = -D0_mat.T @ w
        diag = H0.to_dense().diagonal()
        diag = torch.clamp(diag, min=1e-12)
        return div / diag
    else:
        if isinstance(v, np.ndarray):
            v = ensure_tensor(v, device=str(mesh.device))
        w = H1 @ v
        div = -D0_mat.T @ w
        diag = H0.to_dense().diagonal()
        diag = np.maximum(diag, 1e-12)
        return (div / diag).cpu().numpy()


def curl_scalar(
    mesh: TorchHalfEdgeMesh, u: Union[torch.Tensor, np.ndarray]
) -> Union[torch.Tensor, np.ndarray]:
    """Compute Laplace-Beltrami operator (curl of gradient)."""
    L = laplacian_0(mesh)

    if _HAS_TORCH and isinstance(u, torch.Tensor):
        return L @ u
    else:
        if isinstance(u, np.ndarray):
            u = ensure_tensor(u, device=str(mesh.device))
        return (L @ u).cpu().numpy()


def curl_vector(
    mesh: TorchHalfEdgeMesh, v: Union[torch.Tensor, np.ndarray]
) -> Union[torch.Tensor, np.ndarray]:
    """Compute curl of 1-form (circulation per face).

    Returns face values: d1 @ v
    """
    D1 = d1(mesh)

    if _HAS_TORCH and isinstance(v, torch.Tensor):
        return D1 @ v
    else:
        if isinstance(v, np.ndarray):
            v = ensure_tensor(v, device=str(mesh.device))
        return (D1 @ v).cpu().numpy()


# =============================================================================
# Cache Invalidation (for when mesh changes)
# =============================================================================


def invalidate_caches(mesh: TorchHalfEdgeMesh):
    """Invalidate all cached operators for a mesh."""
    _exterior_cache.invalidate(mesh)
    _hodge_cache.invalidate(mesh)
    _laplacian_cache.invalidate(mesh)


__all__ = [
    "d0",
    "d1",
    "hodge_star_0",
    "hodge_star_1",
    "hodge_star_2",
    "laplacian_0",
    "gradient",
    "divergence",
    "curl_scalar",
    "curl_vector",
    "invalidate_caches",
]
