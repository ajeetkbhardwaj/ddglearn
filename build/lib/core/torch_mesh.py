"""GPU-accelerated mesh data structure with PyTorch.

This module provides a torch-based mesh representation that supports:
- Automatic GPU/CPU device switching
- Cached geometric properties
- Automatic differentiation for gradient-based optimization
- Memory-efficient sparse matrix construction
"""

import numpy as np
import threading
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass, field

try:
    import torch
    from torch import tensor, as_tensor

    _HAS_TORCH = True
except ImportError:
    torch = None
    _HAS_TORCH = False

from core.meshvalid import validate_triangle_mesh
from core.performance import (
    ensure_tensor,
    cached_property,
    face_areas,
    vertex_normals,
    vertex_areas,
)


class HalfEdge:
    """Half-edge data structure element."""

    __slots__ = ("origin", "face", "twin", "next", "prev")

    def __init__(self, origin: int, face: int):
        self.origin: int = origin
        self.face: int = face
        self.twin: Optional[int] = None
        self.next: Optional[int] = None
        self.prev: Optional[int] = None


@dataclass
class MeshCache:
    """Cached geometric data for the mesh."""

    edge_list: Optional[np.ndarray] = None
    edge_map: Optional[dict] = None
    face_areas: Optional[torch.Tensor] = None
    vertex_normals: Optional[torch.Tensor] = None
    vertex_areas: Optional[torch.Tensor] = None
    cotangent_weights: Optional[dict] = None
    adjacency: Optional[List[List[int]]] = None
    boundary_edges: Optional[List[Tuple[int, int]]] = None
    _lock: threading.Lock = field(default_factory=threading.Lock)


class TorchHalfEdgeMesh:
    """GPU-accelerated half-edge mesh for triangle meshes.

    This mesh class supports:
    - Automatic GPU/CPU device management
    - Cached geometric properties
    - Gradient tracking for optimization
    - Memory-efficient operations

    Args:
        vertices: Array of vertex positions, shape (n_vertices, 3)
        faces: Array of face indices, shape (n_faces, 3)
        validate_manifold: Whether to validate mesh manifoldness
        device: Compute device ('cpu', 'cuda', 'mps')
    """

    def __init__(
        self,
        vertices: Union[np.ndarray, torch.Tensor],
        faces: Union[np.ndarray, torch.Tensor],
        validate_manifold: bool = True,
        device: Optional[str] = None,
    ):
        self._device_name = device or "cpu"

        if _HAS_TORCH and isinstance(vertices, torch.Tensor):
            self._vertices = vertices.clone()
        else:
            self._vertices = ensure_tensor(vertices, device=self._device_name)

        if _HAS_TORCH and isinstance(faces, torch.Tensor):
            self._faces = faces.clone()
        else:
            self._faces = ensure_tensor(faces, dtype=torch.long, device=self._device_name)

        if (not _HAS_TORCH or self._device_name == "cpu") and self._faces.shape[1] == 3:
            validate_triangle_mesh(
                self._vertices.cpu().numpy() if _HAS_TORCH else self._vertices,
                self._faces.cpu().numpy() if _HAS_TORCH else self._faces,
            )

        self._n_vertices = self._vertices.shape[0]
        self._n_faces = self._faces.shape[0]

        self.halfedges: List[HalfEdge] = []
        self.vertex_halfedge: List[Optional[int]] = [None] * self._n_vertices
        self._edge_set = set()

        self._cache = MeshCache()

        self._build_halfedges()

        if validate_manifold:
            self._validate_manifold()

    @property
    def device(self) -> torch.device:
        """Get current device."""
        if not _HAS_TORCH:
            return torch.device("cpu")
        return self._vertices.device

    @property
    def vertices(self) -> torch.Tensor:
        """Get vertex positions tensor."""
        return self._vertices

    @vertices.setter
    def vertices(self, value: Union[np.ndarray, torch.Tensor]):
        """Set vertex positions."""
        if _HAS_TORCH and isinstance(value, torch.Tensor):
            self._vertices = value.to(self.device)
        else:
            self._vertices = ensure_tensor(value, device=self._device_name)
        self._invalidate_cache()

    @property
    def faces(self) -> torch.Tensor:
        """Get face indices tensor."""
        return self._faces

    @property
    def n_vertices(self) -> int:
        return self._n_vertices

    @property
    def n_faces(self) -> int:
        return self._n_faces

    @property
    def n_edges(self) -> int:
        if self._cache.edge_list is None:
            self._build_edge_cache()
        return len(self._cache.edge_list)

    @property
    def n_halfedges(self) -> int:
        return len(self.halfedges)

    def _build_halfedges(self):
        """Build half-edge structure from faces."""
        edge_map = {}
        degree = self._faces.shape[1]

        for f_idx in range(self._n_faces):
            f = self._faces[f_idx]
            he_indices = []

            for i in range(degree):
                he = HalfEdge(origin=int(f[i]), face=f_idx)
                he_idx = len(self.halfedges)
                self.halfedges.append(he)
                he_indices.append(he_idx)

            for i in range(degree):
                curr = self.halfedges[he_indices[i]]
                curr.next = he_indices[(i + 1) % degree]
                curr.prev = he_indices[(i - 1) % degree]

                v = curr.origin
                if self.vertex_halfedge[v] is None:
                    self.vertex_halfedge[v] = he_indices[i]

                a = f[i]
                b = f[(i + 1) % 3]
                key = (min(a, b), max(a, b))
                self._edge_set.add(key)

                if key in edge_map:
                    twin_idx = edge_map[key]
                    curr.twin = twin_idx
                    self.halfedges[twin_idx].twin = he_idx
                else:
                    edge_map[key] = he_idx

    def _validate_manifold(self):
        """Validate that mesh is a manifold."""
        edge_count = {}
        degree = self._faces.shape[1]

        f_np = self._faces.cpu().numpy() if _HAS_TORCH else self._faces

        for f in f_np:
            for i in range(degree):
                a = f[i]
                b = f[(i + 1) % degree]
                key = (min(a, b), max(a, b))
                edge_count[key] = edge_count.get(key, 0) + 1

        for key, count in edge_count.items():
            if count > 2:
                raise ValueError(f"Non-manifold edge detected: {key}")

    def _build_edge_cache(self):
        """Build edge list and map cache."""
        processed = set()
        edge_list = []
        edge_map = {}

        for he in self.halfedges:
            a = int(he.origin)
            b = int(self.halfedges[he.next].origin)
            key = (min(a, b), max(a, b))

            if key not in processed:
                processed.add(key)
                edge_map[key] = len(edge_list)
                edge_list.append((a, b))

        self._cache.edge_list = np.array(edge_list)
        self._cache.edge_map = edge_map

    def _invalidate_cache(self):
        """Invalidate cached data when vertices change."""
        with self._cache._lock:
            self._cache.face_areas = None
            self._cache.vertex_normals = None
            self._cache.vertex_areas = None
            self._cache.cotangent_weights = None
            self._cache.adjacency = None

    @cached_property
    def edge_list(self) -> np.ndarray:
        """Get list of unique edges."""
        if self._cache.edge_list is None:
            self._build_edge_cache()
        return self._cache.edge_list

    @cached_property
    def edge_map(self) -> dict:
        """Get edge index map."""
        if self._cache.edge_map is None:
            self._build_edge_cache()
        return self._cache.edge_map

    def outgoing_halfedge(self, v: int) -> Optional[int]:
        """Get first outgoing half-edge from vertex."""
        return self.vertex_halfedge[v]

    def vertex_neighbors(self, v: int) -> List[int]:
        """Get neighboring vertices."""
        if self._cache.adjacency is None:
            self._build_adjacency()
        return self._cache.adjacency[v]

    def _build_adjacency(self):
        """Build vertex adjacency list."""
        self._cache.adjacency = [[] for _ in range(self._n_vertices)]

        for he in self.halfedges:
            a = he.origin
            b = self.halfedges[he.next].origin
            self._cache.adjacency[a].append(b)

        for neighbors in self._cache.adjacency:
            neighbors.sort()
            neighbors = list(dict.fromkeys(neighbors))

    def boundary_edges(self) -> List[Tuple[int, int]]:
        """Get boundary edges."""
        if self._cache.boundary_edges is not None:
            return self._cache.boundary_edges

        boundaries = []
        for he in self.halfedges:
            if he.twin is None:
                a = he.origin
                b = self.halfedges[he.next].origin
                boundaries.append((a, b))

        self._cache.boundary_edges = boundaries
        return boundaries

    @cached_property
    def face_areas_tensor(self) -> torch.Tensor:
        """Compute face areas (cached, GPU-accelerated)."""
        return face_areas(self._vertices, self._faces)

    def face_areas(self) -> np.ndarray:
        """Get face areas as numpy array."""
        if _HAS_TORCH:
            return self.face_areas_tensor.cpu().numpy()
        return self._face_areas_numpy()

    def _face_areas_numpy(self) -> np.ndarray:
        """NumPy fallback for face areas."""
        v0 = self._vertices[self._faces[:, 0]]
        v1 = self._vertices[self._faces[:, 1]]
        v2 = self._vertices[self._faces[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        area = 0.5 * np.linalg.norm(cross, axis=1)
        if self._faces.shape[1] == 4:
            v3 = self._vertices[self._faces[:, 3]]
            area += 0.5 * np.linalg.norm(np.cross(v2 - v0, v3 - v0), axis=1)
        return area

    @cached_property
    def vertex_normals_tensor(self) -> torch.Tensor:
        """Compute vertex normals (cached, GPU-accelerated)."""
        return vertex_normals(self._vertices, self._faces)

    def vertex_normals(self) -> np.ndarray:
        """Get vertex normals as numpy array."""
        if _HAS_TORCH:
            return self.vertex_normals_tensor.cpu().numpy()
        return vertex_normals(self._vertices.cpu().numpy(), self._faces.cpu().numpy())

    @cached_property
    def vertex_area_barycentric_tensor(self) -> torch.Tensor:
        """Compute barycentric vertex areas (cached)."""
        return vertex_areas(self._vertices, self._faces, method="barycentric")

    def vertex_area_barycentric(self) -> np.ndarray:
        """Get barycentric vertex areas."""
        if _HAS_TORCH:
            return self.vertex_area_barycentric_tensor.cpu().numpy()
        return vertex_areas(
            self._vertices.cpu().numpy(), self._faces.cpu().numpy(), method="barycentric"
        )

    @cached_property
    def vertex_area_voronoi_tensor(self) -> torch.Tensor:
        """Compute Voronoi vertex areas (cached)."""
        return vertex_areas(self._vertices, self._faces, method="voronoi")

    def vertex_area_voronoi(self) -> np.ndarray:
        """Get Voronoi vertex areas."""
        if _HAS_TORCH:
            return self.vertex_area_voronoi_tensor.cpu().numpy()
        return vertex_areas(
            self._vertices.cpu().numpy(), self._faces.cpu().numpy(), method="voronoi"
        )

    def to(self, device: Union[str, torch.device]) -> "TorchHalfEdgeMesh":
        """Move mesh to specified device."""
        if not _HAS_TORCH:
            return self

        if isinstance(device, str):
            device = torch.device(device)

        return TorchHalfEdgeMesh(
            self._vertices.to(device),
            self._faces.to(device),
            validate_manifold=False,
            device=device,
        )

    def cpu(self) -> "TorchHalfEdgeMesh":
        """Move mesh to CPU."""
        return self.to("cpu")

    def cuda(self, device: int = 0) -> "TorchHalfEdgeMesh":
        """Move mesh to CUDA."""
        return self.to(f"cuda:{device}")

    def clone(self) -> "TorchHalfEdgeMesh":
        """Create a deep copy of the mesh."""
        return TorchHalfEdgeMesh(
            self._vertices.clone(),
            self._faces.clone(),
            validate_manifold=False,
            device=str(self.device),
        )

    def requires_grad_(self, requires_grad: bool = True):
        """Enable/disable gradient tracking for vertices."""
        if _HAS_TORCH:
            self._vertices.requires_grad_(requires_grad)

    def __repr__(self):
        return f"TorchHalfEdgeMesh(V={self.n_vertices}, F={self.n_faces}, E={self.n_edges}, device={self.device})"


# Alias for backwards compatibility
HalfEdgeMesh = TorchHalfEdgeMesh


__all__ = [
    "TorchHalfEdgeMesh",
    "HalfEdgeMesh",
    "HalfEdge",
]
