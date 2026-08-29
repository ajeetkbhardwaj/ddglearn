# Core API Reference

## Half-Edge Mesh

::: ddglearn.core.halfedge.HalfEdge
    options:
      members: false

::: ddglearn.core.halfedge.HalfEdgeMesh
    options:
      members:
        - n_halfedges
        - n_edges
        - n_vertices
        - edge_list
        - edge_map
        - outgoing_halfedge
        - vertex_neighbors
        - boundary_edges
        - vertex_area_barycentric
        - vertex_area_voronoi

---

## Mesh I/O

::: ddglearn.core.mesh_io.load_mesh

::: ddglearn.core.mesh_io.save_mesh

::: ddglearn.core.mesh_io.load_obj

::: ddglearn.core.mesh_io.save_obj

::: ddglearn.core.mesh_io.load_off

::: ddglearn.core.mesh_io.save_off

::: ddglearn.core.mesh_io.load_ply

::: ddglearn.core.mesh_io.save_ply

::: ddglearn.core.mesh_io.remove_duplicate_vertices

---

## Validation

::: ddglearn.core.meshvalid.validate_triangle_mesh

---

## Performance Utilities

::: ddglearn.core.performance.mesh_hash

::: ddglearn.core.performance.cotangent_weights

::: ddglearn.core.performance.vertex_areas

::: ddglearn.core.performance.face_areas

::: ddglearn.core.performance.vertex_normals

::: ddglearn.core.performance.Timer

::: ddglearn.core.performance.benchmark

::: ddglearn.core.performance.get_system_info

---

## Visualization

::: ddglearn.core.visualization.save_mesh_animation

::: ddglearn.core.visualization.save_mesh_snapshot
