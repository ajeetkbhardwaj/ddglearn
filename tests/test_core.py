"""Tests for core mesh data structures and I/O."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from ddglearn.core import HalfEdgeMesh, load_mesh, save_mesh, validate_triangle_mesh


def test_pyramid_mesh():
    V = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.5],
        ]
    )
    F = np.array(
        [
            [0, 1, 4],
            [1, 2, 4],
            [2, 3, 4],
            [3, 0, 4],
            [0, 1, 2],
            [0, 2, 3],
        ]
    )
    mesh = HalfEdgeMesh(V, F)
    assert mesh.n_vertices == 5
    assert mesh.n_faces == 6
    assert mesh.n_edges == 9


def test_single_triangle():
    V = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    F = np.array([[0, 1, 2]])
    mesh = HalfEdgeMesh(V, F)
    assert mesh.n_vertices == 3
    assert mesh.n_faces == 1


def test_vertex_neighbors():
    V = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
        ]
    )
    F = np.array([[0, 1, 2]])
    mesh = HalfEdgeMesh(V, F)
    neighbors = mesh.vertex_neighbors(0)
    assert len(neighbors) == 1


def test_boundary_edges():
    V = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    F = np.array([[0, 1, 2]])
    mesh = HalfEdgeMesh(V, F)
    boundary = mesh.boundary_edges()
    assert len(boundary) == 3


def test_pentagon_raises():
    V = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.5, 1.5, 0.0], [0.0, 1.0, 0.0]])
    F = np.array([[0, 1, 2, 3, 4]])
    with pytest.raises(ValueError):
        HalfEdgeMesh(V, F)


def test_degenerate_triangle_raises():
    V = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    F = np.array([[0, 1, 2]])
    with pytest.raises(ValueError):
        HalfEdgeMesh(V, F)
        
def test_quad_mesh():
    V = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    F = np.array([[0, 1, 2, 3]])
    mesh = HalfEdgeMesh(V, F)
    assert mesh.n_vertices == 4
    assert mesh.n_faces == 1
    assert mesh.n_edges == 4
    assert mesh.n_halfedges == 4


def test_load_obj(tmp_path):
    obj_content = """# Simple cube
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 1.0 1.0 0.0
v 0.0 1.0 0.0
f 1 2 3
f 1 3 4
"""
    obj_path = tmp_path / "test.obj"
    obj_path.write_text(obj_content)

    V, F = load_mesh(str(obj_path))
    assert V.shape[0] == 4
    assert F.shape[0] == 2
    assert F.shape[1] == 3


def test_save_load_roundtrip(tmp_path):
    V = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    F = np.array([[0, 1, 2]])

    obj_path = tmp_path / "test.obj"
    save_mesh(str(obj_path), V, F)

    V_loaded, F_loaded = load_mesh(str(obj_path))
    assert np.allclose(V, V_loaded)
    assert np.array_equal(F, F_loaded)


def test_validate_valid_mesh():
    V = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    F = np.array([[0, 1, 2]])
    validate_triangle_mesh(V, F)


def test_validate_invalid_indices():
    V = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    F = np.array([[0, 1, 5]])
    with pytest.raises(ValueError):
        validate_triangle_mesh(V, F)


def test_validate_nan_vertices():
    V = np.array([[0.0, 0.0, 0.0], [np.nan, 0.0, 0.0], [0.0, 1.0, 0.0]])
    F = np.array([[0, 1, 2]])
    with pytest.raises(ValueError):
        validate_triangle_mesh(V, F)

def test_validate_disconnected_mesh():
    """Verify the mesh validator gracefully accepts disconnected topological components."""
    V = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
        [10.0, 10.0, 10.0], [11.0, 10.0, 10.0], [10.0, 11.0, 10.0]
    ])
    F = np.array([[0, 1, 2], [3, 4, 5]])
    validate_triangle_mesh(V, F)  # Should not raise

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
