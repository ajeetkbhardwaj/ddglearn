# Core Meshes

The core module provides triangle mesh data structures, I/O, validation, and performance utilities.

## Half-Edge Mesh

```python
from ddglearn.core.halfedge import HalfEdgeMesh
from ddglearn.core.mesh_io import load_mesh

vertices, faces = load_mesh("data/bunny.obj")
mesh = HalfEdgeMesh(vertices, faces)
```

`HalfEdgeMesh` stores triangle connectivity as half-edges, enabling O(1) neighbor traversal:

```python
mesh.n_vertices          # number of vertices
mesh.n_faces             # number of faces
mesh.n_edges             # number of unique edges (cached)
mesh.n_halfedges         # 2 * n_edges

mesh.edge_list           # canonical directed edges: [(i,j), ...]
mesh.edge_map            # {(min,max): edge_index}

mesh.outgoing_halfedge(v)   # half-edge index from vertex v
mesh.vertex_neighbors(v)    # 1-ring neighbor indices
mesh.boundary_edges()       # boundary edge pairs
```

## Vertex Areas

```python
mesh.vertex_area_barycentric()   # (n,) barycentric areas
mesh.vertex_area_voronoi()       # (n,) mixed Voronoi (Meyer et al. 2002)
```

Voronoi areas handle boundary vertices correctly and are used by default in all PDE solvers.

## I/O

```python
from ddglearn.core.mesh_io import load_mesh, save_mesh

# Auto-detect format from extension
vertices, faces = load_mesh("data/bunny.obj")
save_mesh("output.obj", vertices, faces)

# Explicit format
from ddglearn.core.mesh_io import load_obj, load_off, load_ply
vertices, faces = load_obj("data/bunny.obj")
```

Supported formats: OBJ, OFF, PLY (ASCII).

## Validation

```python
from ddglearn.core.meshvalid import validate_triangle_mesh
validate_triangle_mesh(vertices, faces)  # raises ValueError on degenerate triangles
```

## Performance Utilities

```python
from ddglearn.core.performance import (
    cotangent_weights,        # {(i,j): weight} dict
    vertex_areas,             # (n,) barycentric or voronoi
    face_areas,               # (m,)
    vertex_normals,           # (n,3)
    mesh_hash,                # MD5 hash of mesh data
    lru_cache,                # thread-safe cache decorator
    Timer,                    # context manager for timing
    benchmark,                # run function N times, report stats
    get_system_info,          # CPU count, RAM, versions
)
```
