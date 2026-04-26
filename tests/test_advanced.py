"""Tests for advanced modules: PDEs, Spectral, Geometry, and Topology."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents))

from core import HalfEdgeMesh
from tests.test_numerical import load_synthetic

# Import advanced modules
from geometry.decimation import decimate_mesh
from geometry.optimal_transport import wasserstein_barycenter
from pde.cloth import cloth_simulation_step
from pde.wave import simulate_wave
from pde.fluids import fluid_velocity_from_vorticity
from pde.hodge_decomposition import hodge_decomposition, is_divergence_free, is_curl_free
from spectral.wks import compute_wks
from spectral.functional_map import compute_functional_map, point_to_point_from_functional_map
from operators.connection import connection_laplacian, compute_vertex_bases, vertex_holonomy

class TestAdvancedFeatures:
    @pytest.fixture
    def mesh(self):
        return load_synthetic()

    def test_decimate_mesh(self, mesh):
        decimated = decimate_mesh(mesh, grid_resolution=5.0)
        assert isinstance(decimated, HalfEdgeMesh)

    def test_cloth_simulation(self, mesh):
        pos = mesh.vertices.copy()
        vel = np.zeros_like(pos)
        new_pos, new_vel = cloth_simulation_step(mesh, pos, vel, dt=0.01)
        assert new_pos.shape == pos.shape
        assert new_vel.shape == vel.shape

    def test_simulate_wave(self, mesh):
        u0 = np.ones(mesh.n_vertices)
        u_final = simulate_wave(mesh, u0, dt=0.01, steps=3)
        assert u_final.shape == (mesh.n_vertices,)

    def test_fluid_velocity(self, mesh):
        omega = np.random.randn(mesh.n_vertices)
        flux, omega_diffused = fluid_velocity_from_vorticity(mesh, omega, viscosity=0.1, dt=0.01)
        assert flux.shape == (mesh.n_edges,)
        assert omega_diffused.shape == (mesh.n_vertices,)

    def test_hodge_decomposition(self, mesh):
        v_edges = np.random.randn(mesh.n_edges)
        grad_f, div_free, harmonic = hodge_decomposition(mesh, v_edges)
        assert grad_f.shape == (mesh.n_edges,)
        assert div_free.shape == (mesh.n_edges,)
        assert harmonic.shape == (mesh.n_edges,)
        assert is_curl_free(mesh, grad_f, tol=1e-4)
        assert is_divergence_free(mesh, div_free, tol=1e-3)

    def test_wks(self, mesh):
        energies, wks = compute_wks(mesh, k=5)
        assert wks.shape == (mesh.n_vertices, 100)
        assert energies.shape == (100,)

    def test_functional_map(self, mesh):
        desc = np.random.randn(mesh.n_vertices, 3)
        C, evecs1, evecs2 = compute_functional_map(mesh, mesh, desc, desc, k=4)
        assert C.shape == (4, 4)
        p2p = point_to_point_from_functional_map(C, evecs1, evecs2)
        assert p2p.shape == (mesh.n_vertices,)

    def test_connection_laplacian(self, mesh):
        L_conn, bases = connection_laplacian(mesh)
        assert bases.shape == (mesh.n_vertices, 2, 3)
        assert L_conn.shape == (2 * mesh.n_vertices, 2 * mesh.n_vertices)

    def test_vertex_holonomy(self, mesh):
        holonomy = vertex_holonomy(mesh)
        assert holonomy.shape == (mesh.n_vertices,)

    def test_meshedgeconv(self, mesh):
        try:
            import torch
            from core.torch_mesh import TorchHalfEdgeMesh
            from spectral.gnn import MeshEdgeConv
        except ImportError:
            pytest.skip("PyTorch not installed")
            
        t_mesh = TorchHalfEdgeMesh(mesh.vertices, mesh.faces)
        conv = MeshEdgeConv(in_channels=3, out_channels=8).to(t_mesh.device)
        x = torch.randn(t_mesh.n_edges, 3, device=t_mesh.device)
        out = conv(x, t_mesh)
        assert out.shape == (t_mesh.n_edges, 8)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])