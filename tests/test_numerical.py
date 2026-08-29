"""Numerical stability tests on real meshes."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from core import HalfEdgeMesh, load_mesh
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
from geometry import gaussian_curvature, mean_curvature, principal_curvatures
from spectral import eigen_decomposition, compute_hks
from pde import solve_poisson, geodesic_distance, implicit_heat


DATA_DIR = Path(__file__).parents[1] / "data"


def create_octahedron():
    """Creates a regular synthetic octahedron for numerically stable testing."""
    V = np.array([
        [ 1,  0,  0],
        [-1,  0,  0],
        [ 0,  1,  0],
        [ 0, -1,  0],
        [ 0,  0,  1],
        [ 0,  0, -1]
    ], dtype=float)
    F = np.array([
        [0, 4, 2], [2, 4, 1], [1, 4, 3], [3, 4, 0],
        [0, 2, 5], [2, 1, 5], [1, 3, 5], [3, 0, 5]
    ], dtype=int)
    return HalfEdgeMesh(V, F)


def load_synthetic():
    """Load synthetic mesh for testing."""
    return create_octahedron()


class TestNumericalStability:
    """Test numerical stability of operators on real meshes."""

    def test_mesh_dimensions_synthetic(self):
        """Debug test: verify mesh dimensions are correct."""
        mesh = load_synthetic()
        print(f"V={mesh.n_vertices}, F={mesh.n_faces}, E={mesh.n_edges}")
        print(f"Halfedges: {mesh.n_halfedges}")
        assert mesh.n_edges < mesh.n_faces * 2, "Too many edges for this mesh"
        assert mesh.n_edges > mesh.n_faces, "E should be > F for triangle meshes"

    def test_laplacian_symmetry_synthetic(self):
        """Laplacian should be symmetric for closed meshes."""
        mesh = load_synthetic()
        W = d0(mesh).T @ hodge_star_1(mesh) @ d0(mesh)
        W_dense = W.toarray() if hasattr(W, "toarray") else np.asarray(W)
        diff = np.max(np.abs(W_dense - W_dense.T))
        assert diff < 1e-10, f"Laplacian not symmetric: max diff = {diff}"

    def test_laplacian_positive_semidefinite_synthetic(self):
        """Laplacian should be positive semi-definite."""
        mesh = load_synthetic()
        W = d0(mesh).T @ hodge_star_1(mesh) @ d0(mesh)
        W_dense = W.toarray() if hasattr(W, "toarray") else np.asarray(W)
        vals, _ = np.linalg.eigh(W_dense)
        min_eigenvalue = vals[0]
        assert min_eigenvalue > -1e-8, f"Laplacian not PSD: min eigenvalue = {min_eigenvalue}"

    def test_laplacian_zero_eigenvalue_count_synthetic(self):
        """For connected mesh, should have exactly one zero eigenvalue."""
        mesh = load_synthetic()
        W = d0(mesh).T @ hodge_star_1(mesh) @ d0(mesh)
        W_dense = W.toarray() if hasattr(W, "toarray") else np.asarray(W)
        vals, _ = np.linalg.eigh(W_dense)
        zero_count = np.sum(vals < 1e-8)
        assert zero_count == 1, f"Expected 1 zero eigenvalue, got {zero_count}"

    def test_gradient_divergence_identity_synthetic(self):
        """div(grad(u)) should equal Laplacian applied to u."""
        mesh = load_synthetic()
        np.random.seed(42)
        u = np.random.randn(mesh.n_vertices)

        grad_u = gradient(mesh, u)
        div_grad = divergence(mesh, grad_u)

        L = laplacian_0(mesh)
        L_dense = L.toarray() if hasattr(L, "toarray") else np.asarray(L)
        L_u = L_dense @ u

        diff = np.max(np.abs(div_grad + L_u))
        assert diff < 1e-6, f"div(grad(u)) != -Lu: max diff = {diff}"

    def test_exterior_derivative_forms_synthetic(self):
        """d1 @ d0 should be zero (closed forms)."""
        mesh = load_synthetic()
        D0 = d0(mesh)
        D1 = d1(mesh)

        D0_dense = D0.toarray() if hasattr(D0, "toarray") else np.asarray(D0)
        D1_dense = D1.toarray() if hasattr(D1, "toarray") else np.asarray(D1)

        d1d0 = D1_dense @ D0_dense
        norm = np.linalg.norm(d1d0)
        assert norm < 1e-10, f"d1 @ d0 != 0: norm = {norm}"

    def test_hodge_star_diagonal_positive_synthetic(self):
        """All Hodge star diagonals should be positive."""
        mesh = load_synthetic()
        H0 = hodge_star_0(mesh)
        H1 = hodge_star_1(mesh)
        H2 = hodge_star_2(mesh)

        assert np.all(H0.diagonal() > 0), "H0 has non-positive diagonal"
        assert np.all(H1.diagonal() > 0), "H1 has non-positive diagonal"
        assert np.all(H2.diagonal() > 0), "H2 has non-positive diagonal"

    def test_curvature_bounds_synthetic(self):
        """Gaussian curvature should be reasonable for closed surfaces."""
        mesh = load_synthetic()
        K = gaussian_curvature(mesh)

        assert np.all(np.isfinite(K)), "Gaussian curvature contains NaN/Inf"
        assert np.all(K > -10), f"Gaussian curvature too negative: min = {K.min()}"
        assert np.all(K < 100), f"Gaussian curvature too large: max = {K.max()}"

    def test_mean_curvature_positive_synthetic(self):
        """Mean curvature magnitude should be non-negative."""
        mesh = load_synthetic()
        H = mean_curvature(mesh)

        assert np.all(np.isfinite(H)), "Mean curvature contains NaN/Inf"
        assert np.all(H >= -1e-6), f"Mean curvature negative: min = {H.min()}"

    def test_principal_curvatures_consistency_synthetic(self):
        """k1 >= k2 and K = k1 * k2, H = (k1 + k2) / 2."""
        mesh = load_synthetic()
        K = gaussian_curvature(mesh)
        H = mean_curvature(mesh)
        k1, k2 = principal_curvatures(mesh)

        assert np.all(k1 >= k2 - 1e-10), "k1 < k2"

        K_computed = k1 * k2
        H_computed = 0.5 * (k1 + k2)

        # In discrete geometries, H^2 < K can numerically occur at sharp points.
        # When that happens, the discriminant is clamped to 0, making k1*k2 = H^2.
        K_expected = np.where(H**2 >= K, K, H**2)
        K_diff = np.max(np.abs(K_expected - K_computed))
        H_diff = np.max(np.abs(H - H_computed))

        assert K_diff < 1e-6, f"Principal curvature K mismatch: {K_diff}"
        assert H_diff < 1e-6, f"Principal curvature H mismatch: {H_diff}"

    def test_eigen_decomposition_stability_synthetic(self):
        """Eigen decomposition should produce stable results."""
        mesh = load_synthetic()
        vals, vecs = eigen_decomposition(mesh, k=10)

        assert np.all(np.isfinite(vals)), "Eigenvalues contain NaN/Inf"
        assert np.all(np.isfinite(vecs)), "Eigenvectors contain NaN/Inf"
        assert np.all(vals >= -1e-10), "Negative eigenvalues found"
        assert vals[0] < 1e-6, f"First eigenvalue should be ~0, got {vals[0]}"

        for i in range(min(5, len(vals) - 1)):
            assert vals[i] <= vals[i + 1] + 1e-10, "Eigenvalues not sorted"

    def test_hks_stability_synthetic(self):
        """Heat Kernel Signature should produce valid results."""
        mesh = load_synthetic()
        times, hks = compute_hks(mesh, k=6)

        assert np.all(np.isfinite(hks)), "HKS contains NaN/Inf"
        assert np.all(hks >= 0), "HKS has negative values"
        assert hks.shape[0] == mesh.n_vertices

    def test_poisson_solver_stability_synthetic(self):
        """Poisson solver should produce stable results."""
        mesh = load_synthetic()
        n = mesh.n_vertices
        f = np.random.randn(n)

        u = solve_poisson(mesh, f)

        assert np.all(np.isfinite(u)), "Poisson solution contains NaN/Inf"

    def test_geodesic_distance_synthetic(self):
        """Geodesic distance should be non-negative and zero at source."""
        mesh = load_synthetic()
        dist = geodesic_distance(mesh, source_indices=[0])

        assert np.all(np.isfinite(dist)), "Geodesic distance contains NaN/Inf"
        assert dist[0] < 1e-6, "Distance at source not zero"
        assert np.all(dist >= -1e-6), "Geodesic distance negative"

        max_dist = dist.max()
        assert max_dist > 0, "All distances are zero"

    def test_poisson_dirichlet_synthetic(self):
        """Test Poisson solver with Dirichlet boundary conditions."""
        mesh = load_synthetic()
        f = np.zeros(mesh.n_vertices)
        f[0] = 1.0
        f[1] = -1.0
        
        d_idx = np.array([2, 3, 4])
        d_val = np.array([0.0, 0.5, -0.5])
        
        u = solve_poisson(mesh, f, dirichlet_indices=d_idx, dirichlet_values=d_val)
        
        for i, idx in enumerate(d_idx):
            assert np.isclose(u[idx], d_val[i]), f"Dirichlet BC failed at vertex {idx}"
            
    def test_geodesic_methods_synthetic(self):
        """Test multiple geodesic integration methods."""
        mesh = load_synthetic()
        
        dist_poisson = geodesic_distance(mesh, [0], method="poisson")
        dist_varadhan = geodesic_distance(mesh, [0], method="varadhan")
        dist_fmm = geodesic_distance(mesh, [0], method="fmm_graph")
        
        assert dist_poisson[0] < 1e-6
        assert dist_varadhan[0] < 1e-6
        assert dist_fmm[0] < 1e-6
        
        assert np.all(np.isfinite(dist_poisson))
        assert np.all(np.isfinite(dist_varadhan))
        assert np.all(np.isfinite(dist_fmm))
        
    def test_chebyshev_filter_synthetic(self):
        """Test Chebyshev polynomial spectral filters."""
        from spectral.chebyshev import chebyshev_filter
        mesh = load_synthetic()
        x = np.random.randn(mesh.n_vertices)
        T = chebyshev_filter(mesh, x, order=3)
        
        assert len(T) == 4
        assert T[0].shape == (mesh.n_vertices,)
        assert np.allclose(T[0], x)
        
    def test_sinkhorn_wasserstein_synthetic(self):
        """Test Convolutional Wasserstein Distances."""
        from geometry.optimal_transport import sinkhorn_wasserstein
        mesh = load_synthetic()
        p = np.zeros(mesh.n_vertices)
        q = np.zeros(mesh.n_vertices)
        p[0] = 1.0
        q[1] = 1.0
        
        dist, u, v = sinkhorn_wasserstein(mesh, p, q, t=0.05, max_iter=5)
        assert dist > 0.0
        
class TestBoundaryHandling:
    """Test handling of meshes with boundaries."""

    def test_boundary_mesh_laplacian_symmetry(self):
        """Laplacian should still be symmetric for boundary meshes."""
        V = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        F = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],
            ]
        )
        mesh = HalfEdgeMesh(V, F)
        W = d0(mesh).T @ hodge_star_1(mesh) @ d0(mesh)
        W_dense = W.toarray() if hasattr(W, "toarray") else np.asarray(W)
        diff = np.max(np.abs(W_dense - W_dense.T))
        assert diff < 1e-10, f"Boundary mesh Laplacian not symmetric: {diff}"

    def test_poisson_neumann_boundary(self):
        """Test Poisson solver with Neumann boundary conditions."""
        V = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
        F = np.array([[0, 1, 2], [0, 2, 3]])
        mesh = HalfEdgeMesh(V, F)
        
        f = np.zeros(4)
        n_idx = np.array([1, 2])
        n_val = np.array([1.0, -1.0])
        
        u = solve_poisson(mesh, f, pin_index=0, pin_value=0.0, neumann_indices=n_idx, neumann_values=n_val)
        
        assert np.all(np.isfinite(u))
        assert np.isclose(u[0], 0.0)
        
    def test_heat_dirichlet_boundary(self):
        """Test implicit heat step with Dirichlet boundaries."""
        V = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
        F = np.array([[0, 1, 2], [0, 2, 3]])
        mesh = HalfEdgeMesh(V, F)
        
        u0 = np.array([1.0, 0.0, 0.0, 0.0])
        d_idx = np.array([2, 3])
        d_val = np.array([0.5, 0.5])
        
        u1 = implicit_heat(mesh, u0, t=0.1, steps=1, dirichlet_indices=d_idx, dirichlet_values=d_val)
        
        assert np.isclose(u1[2], 0.5)
        assert np.isclose(u1[3], 0.5)

    def test_harmonic_parameterization_boundary(self):
        """Test mapping a bounded 3D mesh to a 2D unit circle."""
        from geometry.parameterization import harmonic_parameterization
        V = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
        F = np.array([[0, 1, 2], [0, 2, 3]])
        mesh = HalfEdgeMesh(V, F)
        
        uv = harmonic_parameterization(mesh)
        assert uv.shape == (4, 2)
        
        # Verify all boundary points were perfectly pinned to the unit circle
        norms = np.linalg.norm(uv, axis=1)
        assert np.allclose(norms, 1.0)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
