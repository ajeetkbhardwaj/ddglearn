import numpy as np
from typing import List, Optional, Tuple
from .meshvalid import validate_triangle_mesh


class HalfEdge:
    __slots__ = ("origin", "face", "twin", "next", "prev")

    def __init__(self, origin: int, face: int):
        self.origin: int = origin
        self.face: int = face
        self.twin: Optional[int] = None
        self.next: Optional[int] = None
        self.prev: Optional[int] = None


class HalfEdgeMesh:
    """
    Robust half-edge mesh for triangle meshes.

    Designed for DEC operators and geometric processing.
    The mesh stores vertices as an (n_vertices, 3) array and faces
    as an (n_faces, 3) array of integer indices.

    Attributes:
        vertices: Array of vertex positions, shape (n_vertices, 3).
        faces: Array of face indices, shape (n_faces, 3).
        n_vertices: Number of vertices.
        n_faces: Number of faces.
        n_edges: Number of unique edges.
        n_halfedges: Number of half-edges (2 * n_edges for closed meshes).
    """

    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        validate_manifold: bool = True,
    ) -> None:
        self.vertices = np.asarray(vertices, dtype=float)
        self.faces = np.asarray(faces, dtype=int)

        if self.faces.ndim != 2 or self.faces.shape[1] not in (3, 4):
            raise ValueError("Only triangle (m x 3) or quad (m x 4) meshes supported.")

        self.n_vertices = self.vertices.shape[0]
        self.n_faces = self.faces.shape[0]

        if self.faces.shape[1] == 3:
            validate_triangle_mesh(self.vertices, self.faces)

        self.halfedges: List[HalfEdge] = []
        self.vertex_halfedge: List[Optional[int]] = [None] * self.n_vertices

        self._build_halfedges()
        self._cache = {}

        if validate_manifold:
            self._validate_manifold()

    def _build_halfedges(self):
        """Vectorized construction of the half-edge data structure."""
        n_v = self.n_vertices
        n_f = self.n_faces
        degree = self.faces.shape[1]
        n_he_total = n_f * degree

        # Pre-allocate half-edges
        self.halfedges = [None] * n_he_total
        
        # 1. Create half-edges and basic connectivity (next, prev, face, origin)
        for f_idx in range(n_f):
            for i in range(degree):
                he_idx = f_idx * degree + i
                he = HalfEdge(origin=int(self.faces[f_idx, i]), face=f_idx)
                he.next = f_idx * degree + (i + 1) % degree
                he.prev = f_idx * degree + (i - 1) % degree
                self.halfedges[he_idx] = he
                
                # Set vertex's outgoing half-edge
                if self.vertex_halfedge[he.origin] is None:
                    self.vertex_halfedge[he.origin] = he_idx

        # 2. Vectorized twin finding
        # Edges as (start, end) pairs
        e_start = self.faces.flatten()
        e_end = np.roll(self.faces, -1, axis=1).flatten()
        
        # Sort each edge to get undirected keys for hashing
        # key = min(u, v) * n_v + max(u, v)
        v_min = np.minimum(e_start, e_end)
        v_max = np.maximum(e_start, e_end)
        e_hash = v_min.astype(np.int64) * n_v + v_max.astype(np.int64)
        
        # Sort hashes to find pairs
        sort_idx = np.argsort(e_hash)
        sorted_hash = e_hash[sort_idx]
        
        # Find adjacent identical hashes (potential twins)
        matches = sorted_hash[:-1] == sorted_hash[1:]
        
        # Potential twins are at sort_idx[i] and sort_idx[i+1]
        idx1 = sort_idx[:-1][matches]
        idx2 = sort_idx[1:][matches]
        
        # Set twins
        for i1, i2 in zip(idx1, idx2):
            self.halfedges[i1].twin = int(i2)
            self.halfedges[i2].twin = int(i1)

    def _validate_manifold(self):
        edge_count = {}
        degree = self.faces.shape[1]
        for f in self.faces:
            for i in range(degree):
                a = f[i]
                b = f[(i + 1) % degree]
                key = (min(a, b), max(a, b))
                edge_count[key] = edge_count.get(key, 0) + 1

        for key, count in edge_count.items():
            if count > 2:
                raise ValueError(f"Non-manifold edge detected: {key}")

    @property
    def n_halfedges(self) -> int:
        return len(self.halfedges)

    @property
    def n_edges(self) -> int:
        if hasattr(self, "_cached_n_edges"):
            return self._cached_n_edges

        edge_set = set()
        for he in self.halfedges:
            a = he.origin
            b = self.halfedges[he.next].origin
            key = (min(a, b), max(a, b))
            edge_set.add(key)
        self._cached_n_edges = len(edge_set)
        return self._cached_n_edges

    def outgoing_halfedge(self, v: int) -> Optional[int]:
        return self.vertex_halfedge[v]

    def vertex_neighbors(self, v: int) -> List[int]:
        start = self.vertex_halfedge[v]
        if start is None:
            return []

        neighbors = []
        he_idx = start

        while True:
            he = self.halfedges[he_idx]
            nxt = self.halfedges[he.next]
            neighbors.append(nxt.origin)

            if he.twin is None:
                break

            he_idx = self.halfedges[he.twin].next
            if he_idx == start:
                break

        return neighbors

    def boundary_edges(self) -> List[Tuple[int, int]]:
        edges = []
        for he in self.halfedges:
            if he.twin is None:
                a = he.origin
                b = self.halfedges[he.next].origin
                edges.append((a, b))
        return edges

    @property
    def edge_list(self) -> List[Tuple[int, int]]:
        if "edge_list" not in self._cache:
            from ..operators.exterior_derivative import edge_list_and_map
            self._cache["edge_list"], self._cache["edge_map"] = edge_list_and_map(self)
        return self._cache["edge_list"]

    @property
    def edge_map(self) -> dict:
        if "edge_map" not in self._cache:
            from ..operators.exterior_derivative import edge_list_and_map
            self._cache["edge_list"], self._cache["edge_map"] = edge_list_and_map(self)
        return self._cache["edge_map"]

    def vertex_area_barycentric(self) -> np.ndarray:
        V = self.vertices
        F = self.faces
        degree = F.shape[1]

        areas = np.zeros(self.n_vertices)
        
        if degree == 3:
            v0 = V[F[:, 0]]
            v1 = V[F[:, 1]]
            v2 = V[F[:, 2]]
            face_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
            for k in range(3):
                np.add.at(areas, F[:, k], face_areas / 3.0)
        else:
            v0, v1, v2, v3 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]], V[F[:, 3]]
            area1 = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
            area2 = 0.5 * np.linalg.norm(np.cross(v2 - v0, v3 - v0), axis=1)
            face_areas = area1 + area2
            for k in range(4):
                np.add.at(areas, F[:, k], face_areas / 4.0)

        return areas

    def vertex_area_voronoi(self) -> np.ndarray:
        """
        Compute mixed Voronoi area per vertex (Meyer et al. 2002).

        For interior vertices: uses true mixed Voronoi areas with cotangent weights.
        For boundary vertices: uses half the incident face areas (clipped Voronoi).

        This ensures the Laplacian is symmetric for both closed and boundary meshes.

        Returns array of shape (n_vertices,)
        """
        if self.faces.shape[1] == 4:
            return self.vertex_area_barycentric()
            
        V = self.vertices
        F = self.faces
        n_v = self.n_vertices
        n_f = self.n_faces

        boundary_verts = set()
        for a, b in self.boundary_edges():
            boundary_verts.add(a)
            boundary_verts.add(b)

        face_areas = np.zeros(n_f, dtype=float)
        v0 = V[F[:, 0]]
        v1 = V[F[:, 1]]
        v2 = V[F[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        face_areas = 0.5 * np.linalg.norm(cross, axis=1)

        A = np.zeros(n_v, dtype=float)

        # Vectorized per-face cotangent and area computation.
        def _cot_vec(ba, ca):
            cr = np.cross(ba, ca)
            nm = np.linalg.norm(cr, axis=1)
            return np.where(nm < 1e-12, 0.0, np.sum(ba * ca, axis=1) / np.maximum(nm, 1e-12))

        cot_i = _cot_vec(v1 - v0, v2 - v0)
        cot_j = _cot_vec(v0 - v1, v2 - v1)
        cot_k = _cot_vec(v0 - v2, v1 - v2)

        e_ij_len_sq = np.sum((v1 - v0) ** 2, axis=1)
        e_jk_len_sq = np.sum((v2 - v1) ** 2, axis=1)
        e_ki_len_sq = np.sum((v0 - v2) ** 2, axis=1)

        is_boundary = np.zeros(n_v, dtype=bool)
        for bv in boundary_verts:
            is_boundary[bv] = True

        bnd_i = is_boundary[F[:, 0]]
        bnd_j = is_boundary[F[:, 1]]
        bnd_k = is_boundary[F[:, 2]]

        c0 = np.where(bnd_i, face_areas * 0.5,
                       0.125 * (cot_k * e_ij_len_sq + cot_j * e_ki_len_sq))
        c1 = np.where(bnd_j, face_areas * 0.5,
                       0.125 * (cot_i * e_jk_len_sq + cot_k * e_ij_len_sq))
        c2 = np.where(bnd_k, face_areas * 0.5,
                       0.125 * (cot_j * e_ki_len_sq + cot_i * e_jk_len_sq))

        np.add.at(A, F[:, 0], c0)
        np.add.at(A, F[:, 1], c1)
        np.add.at(A, F[:, 2], c2)

        A = np.maximum(A, 1e-12)
        return A

    def __repr__(self):
        return f"HalfEdgeMesh(V={self.n_vertices}, F={self.n_faces}, E={self.n_edges})"
