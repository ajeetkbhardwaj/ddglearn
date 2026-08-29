"""Core mesh data structures and I/O."""

from .halfedge import HalfEdgeMesh, HalfEdge
from .meshvalid import validate_triangle_mesh
from .mesh_io import (
    load_mesh, save_mesh, load_obj, save_obj, 
    load_off, save_off, load_ply, save_ply, remove_duplicate_vertices
)

__all__ = [
    "HalfEdgeMesh", "HalfEdge",
    "validate_triangle_mesh", "load_mesh", "save_mesh",
    "load_obj", "save_obj", "load_off", "save_off",
    "load_ply", "save_ply", "remove_duplicate_vertices"
]