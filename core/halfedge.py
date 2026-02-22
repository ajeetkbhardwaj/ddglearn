
import numpy as np
from typing import List, Optional, Tuple
from core.meshvalid import validate_triangle_mesh
from operators.exterior_derivative import edge_list_and_map

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
    """

    def __init__(self, vertices, faces, validate_manifold=True):
        self.vertices = np.asarray(vertices, dtype=float)
        self.faces = np.asarray(faces, dtype=int)

        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError("Only triangle meshes supported (m x 3 faces).")
 
        self.n_vertices = self.vertices.shape[0]
        self.n_faces = self.faces.shape[0]

        validate_triangle_mesh(self.vertices, self.faces)
        # --------------------------------------------------------- 

        self.halfedges: List[HalfEdge] = []
        self.vertex_halfedge: List[Optional[int]] = [None] * self.n_vertices

        self._edge_set = set()
        self._build_halfedges()

        if validate_manifold:
            self._validate_manifold()

    # ------------------------------------------------------------------ #
    # Build structure
    # ------------------------------------------------------------------ #

    def _build_halfedges(self):
        edge_map = {}

        for f_idx, f in enumerate(self.faces):
            he_indices = []

            # Create 3 halfedges
            for i in range(3):
                he = HalfEdge(origin=int(f[i]), face=f_idx)
                he_idx = len(self.halfedges)
                self.halfedges.append(he)
                he_indices.append(he_idx)

            # Link them cyclically
            for i in range(3):
                curr = self.halfedges[he_indices[i]]
                curr.next = he_indices[(i + 1) % 3]
                curr.prev = he_indices[(i - 1) % 3]

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
                    self.halfedges[twin_idx].twin = he_indices[i]
                else:
                    edge_map[key] = he_indices[i]

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def _validate_manifold(self):
        # Non-manifold edge check
        edge_count = {}
        for f in self.faces:
            for i in range(3):
                a = f[i]
                b = f[(i + 1) % 3]
                key = (min(a, b), max(a, b))
                edge_count[key] = edge_count.get(key, 0) + 1

        for key, count in edge_count.items():
            if count > 2:
                raise ValueError(f"Non-manifold edge detected: {key}")

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @property
    def n_halfedges(self) -> int:
        return len(self.halfedges)

    @property
    def n_edges(self) -> int:
        return len(self._edge_set)

    # ------------------------------------------------------------------ #
    # Vertex queries
    # ------------------------------------------------------------------ #

    def outgoing_halfedge(self, v: int) -> Optional[int]:
        return self.vertex_halfedge[v]

    def vertex_neighbors(self, v: int) -> List[int]:
        """
        Proper circulation around vertex.
        Works for interior and boundary vertices.
        """
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
                break  # boundary reached

            he_idx = self.halfedges[he.twin].next
            if he_idx == start:
                break

        return neighbors

    # ------------------------------------------------------------------ #
    # Boundary
    # ------------------------------------------------------------------ #

    def boundary_edges(self) -> List[Tuple[int, int]]:
        edges = []
        for he in self.halfedges:
            if he.twin is None:
                a = he.origin
                b = self.halfedges[he.next].origin
                edges.append((a, b))
        return edges

    # ------------------------------------------------------------------ #
    # Areas
    # ------------------------------------------------------------------ #

    def vertex_area_barycentric(self) -> np.ndarray:
        """
        Barycentric vertex area (1/3 of incident face areas).
        Stable for DEC Laplacian.
        """
        V = self.vertices
        F = self.faces

        areas = np.zeros(self.n_vertices)

        v0 = V[F[:, 0]]
        v1 = V[F[:, 1]]
        v2 = V[F[:, 2]]

        face_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)

        for k in range(3):
            np.add.at(areas, F[:, k], face_areas / 3.0)

        return areas
    def vertex_area_voronoi(self) -> np.ndarray:
        """
        Compute mixed Voronoi area per vertex (Meyer et al. 2002).
        For now: use barycentric (1/3 per adjacent face) area as stable fallback.
        Returns array of shape (n_vertices,)
        """
        V = self.vertices
        F = self.faces
        n_v = self.n_vertices

        A = np.zeros(n_v, dtype=float)

        for f in F:
            i, j, k = [int(x) for x in f]
            v0, v1, v2 = V[i], V[j], V[k]

            face_area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))

            A[i] += face_area / 3.0
            A[j] += face_area / 3.0
            A[k] += face_area / 3.0

        return A

    # ------------------------------------------------------------------ #

    def __repr__(self):
        return f"HalfEdgeMesh(V={self.n_vertices}, F={self.n_faces}, E={self.n_edges})"
    
"""



import numpy as np
from typing import List, Optional


class HalfEdge:
    __slots__ = ("origin", "face", "twin", "next", "prev")

    def __init__(self, origin: int, face: int):
        self.origin: int = origin
        self.face: int = face
        self.twin: Optional[int] = None
        self.next: Optional[int] = None
        self.prev: Optional[int] = None


class HalfEdgeMesh:
    

    def __init__(self, vertices, faces):
        self.vertices = np.asarray(vertices, dtype=float)
        self.faces = np.asarray(faces, dtype=int)
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError("Only triangle meshes (m x 3 faces) are supported")

        self.n_vertices = int(self.vertices.shape[0])
        self.n_faces = int(self.faces.shape[0])

        self.halfedges: List[HalfEdge] = []
        self.vertex_halfedge: List[Optional[int]] = [None] * self.n_vertices

        self._build_halfedges()

    def _build_halfedges(self):
        edge_map = {}
        for f_idx, f in enumerate(self.faces):
            he_inds = []
            for i in range(3):
                v = int(f[i])
                he = HalfEdge(origin=v, face=f_idx)
                he_idx = len(self.halfedges)
                self.halfedges.append(he)
                he_inds.append(he_idx)

            for i in range(3):
                curr_idx = he_inds[i]
                next_idx = he_inds[(i + 1) % 3]
                prev_idx = he_inds[(i - 1) % 3]
                curr = self.halfedges[curr_idx]
                curr.next = next_idx
                curr.prev = prev_idx

                if self.vertex_halfedge[curr.origin] is None:
                    self.vertex_halfedge[curr.origin] = curr_idx

                a = int(self.faces[f_idx][i])
                b = int(self.faces[f_idx][(i + 1) % 3])
                key = (min(a, b), max(a, b))
                if key in edge_map:
                    twin_idx = edge_map[key]
                    curr.twin = twin_idx
                    self.halfedges[twin_idx].twin = curr_idx
                else:
                    edge_map[key] = curr_idx

    @property
    def n_halfedges(self) -> int:
        return len(self.halfedges)

    @property
    def n_edges(self) -> int:
        if not hasattr(self, '_n_edges_cached'):
            # Count unique edges by deduplicating on (min(a,b), max(a,b))
            edge_set = set()
            for he_idx, he in enumerate(self.halfedges):
                a = int(he.origin)
                b = int(self.halfedges[he.next].origin)
                key = (min(a, b), max(a, b))
                edge_set.add(key)
            self._n_edges_cached = len(edge_set)
        return self._n_edges_cached

    def outgoing_halfedge(self, v: int) -> Optional[int]:
        return self.vertex_halfedge[v]

    def vertex_neighbors(self, v: int) -> List[int]:
        start = self.vertex_halfedge[v]
        if start is None:
            return []
        neigh = []
        he_idx = start
        while True:
            he = self.halfedges[he_idx]
            next_he = self.halfedges[he.next]
            neigh.append(next_he.origin)
            # walk to the previous twin around the vertex
            twin = self.halfedges[he.prev].twin
            if twin is None:
                break
            he_idx = twin
            if he_idx == start:
                break
        return neigh

    def boundary_edges(self) -> List[tuple]:
        edges = []
        for idx, he in enumerate(self.halfedges):
            if he.twin is None:
                a = he.origin
                b = self.halfedges[he.next].origin
                edges.append((a, b))
        return edges

    def vertex_area_voronoi(self) -> np.ndarray:
        V = self.vertices
        F = self.faces

        areas = np.zeros(self.n_vertices)

        v0 = V[F[:,0]]
        v1 = V[F[:,1]]
        v2 = V[F[:,2]]

        face_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)

        for k in range(3):
            np.add.at(areas, F[:,k], face_areas / 3.0)

        return areas

    def __repr__(self):
        return f"HalfEdgeMesh(V={self.n_vertices}, F={self.n_faces}, HE={self.n_halfedges})"
"""