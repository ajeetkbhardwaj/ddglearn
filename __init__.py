"""DDG: Discrete Differential Geometry Library.

A comprehensive, production-ready DEC/FEM implementation for triangle meshes
with spectral geometry, curvature analysis, PDE solvers, and vector field processing.

Quick Start:
    from ddg import HalfEdgeMesh, load_mesh
    from ddg.geometry import gaussian_curvature
    from ddg.operators import laplacian_0
    from ddg.pde import solve_poisson
    from ddg.spectral import eigen_decomposition
    from ddg.visualization import plot_scalar_field
"""

__version__ = "0.1.0"

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import (
    HalfEdgeMesh,
    load_mesh,
    save_mesh,
    load_obj,
    save_obj,
    load_off,
    save_off,
    load_ply,
    save_ply,
    validate_triangle_mesh,
)

from geometry import (
    gaussian_curvature,
    mean_curvature,
    mean_curvature_vector,
    principal_curvatures,
    compute_vertex_normals,
    shape_operator_tensor,
    principal_directions,
)

from geometry.optimal_transport import sinkhorn_wasserstein, wasserstein_barycenter
from geometry.decimation import decimate_mesh
from geometry.parameterization import harmonic_parameterization

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

from operators.cotangent_laplacian import cotangent_laplacian

from operators.connection import (
    connection_laplacian,
    compute_vertex_bases,
    vertex_holonomy,
)

from pde import (
    solve_poisson,
    implicit_heat,
    implicit_heat_step,
    simulate_wave,
    wave_step,
    geodesic_distance,
    heat_method_geodesics,
    solve_heat_diffusion,
    hodge_decomposition,
    hodge_star_decomposition,
    is_divergence_free,
    is_curl_free,
)

from pde.fluids import fluid_velocity_from_vorticity
from pde.cloth import cloth_simulation_step

from spectral import (
    eigen_decomposition,
    compute_hks,
    compute_wks,
)

from spectral.chebyshev import chebyshev_filter, scaled_laplacian

from visualization import (
    initialize_viewer,
    plot_scalar_field,
    plot_vector_field,
    plot_curvature,
    plot_normals,
    plot_principal_directions,
    plot_spectral_modes,
    plot_hks,
    plot_geodesic_distance,
    batch_visualization,
)

__all__ = [
    "__version__",
    # core
    "HalfEdgeMesh",
    "load_mesh",
    "save_mesh",
    "load_obj",
    "save_obj",
    "load_off",
    "save_off",
    "load_ply",
    "save_ply",
    "validate_triangle_mesh",
    # geometry
    "gaussian_curvature",
    "mean_curvature",
    "mean_curvature_vector",
    "principal_curvatures",
    "compute_vertex_normals",
    "shape_operator_tensor",
    "principal_directions",
    "sinkhorn_wasserstein",
    "wasserstein_barycenter",
    "decimate_mesh",
    "harmonic_parameterization",
    # operators
    "d0",
    "d1",
    "hodge_star_0",
    "hodge_star_1",
    "hodge_star_2",
    "laplacian_0",
    "cotangent_laplacian",
    "connection_laplacian",
    "compute_vertex_bases",
    "vertex_holonomy",
    "gradient",
    "divergence",
    "curl_scalar",
    "curl_vector",
    # pde
    "solve_poisson",
    "implicit_heat",
    "implicit_heat_step",
    "simulate_wave",
    "wave_step",
    "geodesic_distance",
    "heat_method_geodesics",
    "solve_heat_diffusion",
    "hodge_decomposition",
    "hodge_star_decomposition",
    "is_divergence_free",
    "is_curl_free",
    # spectral
    "eigen_decomposition",
    "compute_hks",
    "compute_wks",
    "chebyshev_filter",
    "scaled_laplacian",
    # visualization
    "initialize_viewer",
    "plot_scalar_field",
    "plot_vector_field",
    "plot_curvature",
    "plot_normals",
    "plot_principal_directions",
    "plot_spectral_modes",
    "plot_hks",
    "plot_geodesic_distance",
    "batch_visualization",
]
