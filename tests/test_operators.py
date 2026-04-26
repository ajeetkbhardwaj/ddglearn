"""Tests for DEC operators."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from core import HalfEdgeMesh
from operators import (
    d0,
    d1,
    hodge_star_0,
    hodge_star_1,
    hodge_star_2,
    laplacian_0,
    gradient,
    divergence,
    curl_scalar,
    curl_vector,
)


@pytest.fixture
def simple_mesh():
    V = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.5],
        ]
    )
    F = np.array([[0, 1, 3], [1, 2, 3], [2, 0, 3]])
    return HalfEdgeMesh(V, F)


class TestExteriorDerivative:
    def test_d0_shape(self, simple_mesh):
        D0 = d0(simple_mesh)
        assert D0.shape[0] == simple_mesh.n_edges
        assert D0.shape[1] == simple_mesh.n_vertices

    def test_d1_shape(self, simple_mesh):
        D1 = d1(simple_mesh)
        assert D1.shape[0] == simple_mesh.n_faces
        assert D1.shape[1] == simple_mesh.n_edges


class TestHodgeStar:
    def test_hodge_star_0_shape(self, simple_mesh):
        H0 = hodge_star_0(simple_mesh)
        assert H0.shape == (simple_mesh.n_vertices, simple_mesh.n_vertices)

    def test_hodge_star_1_shape(self, simple_mesh):
        H1 = hodge_star_1(simple_mesh)
        assert H1.shape == (simple_mesh.n_edges, simple_mesh.n_edges)

    def test_hodge_star_2_shape(self, simple_mesh):
        H2 = hodge_star_2(simple_mesh)
        assert H2.shape == (simple_mesh.n_faces, simple_mesh.n_faces)

    def test_hodge_star_0_positive(self, simple_mesh):
        H0 = hodge_star_0(simple_mesh)
        diag = H0.diagonal()
        assert np.all(diag > 0)


class TestLaplacian:
    def test_laplacian_shape(self, simple_mesh):
        L = laplacian_0(simple_mesh)
        assert L.shape == (simple_mesh.n_vertices, simple_mesh.n_vertices)

    def test_laplacian_symmetric(self, simple_mesh):
        W = d0(simple_mesh).T @ hodge_star_1(simple_mesh) @ d0(simple_mesh)
        W_dense = W.toarray() if hasattr(W, "toarray") else np.asarray(W)
        assert np.allclose(W_dense, W_dense.T, rtol=1e-10, atol=1e-12)

    def test_laplacian_positive_definite(self, simple_mesh):
        W = d0(simple_mesh).T @ hodge_star_1(simple_mesh) @ d0(simple_mesh)
        W_dense = W.toarray() if hasattr(W, "toarray") else np.asarray(W)
        vals, vecs = np.linalg.eigh(W_dense)
        assert vals[0] >= -1e-10


class TestGradientDivergence:
    def test_gradient_shape(self, simple_mesh):
        u = np.ones(simple_mesh.n_vertices)
        grad_u = gradient(simple_mesh, u)
        assert grad_u.shape[0] == simple_mesh.n_edges

    def test_divergence_shape(self, simple_mesh):
        v = np.ones(simple_mesh.n_edges)
        div_v = divergence(simple_mesh, v)
        assert div_v.shape[0] == simple_mesh.n_vertices


class TestCurl:
    def test_curl_scalar_shape(self, simple_mesh):
        u = np.ones(simple_mesh.n_vertices)
        curl_u = curl_scalar(simple_mesh, u)
        assert curl_u.shape[0] == simple_mesh.n_vertices

    def test_curl_vector_shape(self, simple_mesh):
        v = np.ones(simple_mesh.n_edges)
        curl_v = curl_vector(simple_mesh, v)
        assert curl_v.shape[0] == simple_mesh.n_faces


class TestConsistency:
    def test_gradient_divergence_identity(self, simple_mesh):
        u = np.random.randn(simple_mesh.n_vertices)
        grad_u = gradient(simple_mesh, u)
        div_grad = divergence(simple_mesh, grad_u)
        L = laplacian_0(simple_mesh)
        L_dense = L.toarray() if hasattr(L, "toarray") else np.asarray(L)
        L_u_dense = L_dense.dot(u)
        assert np.allclose(div_grad, -L_u_dense, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
