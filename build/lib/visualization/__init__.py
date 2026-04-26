"""Visualization helpers using Polyscope."""

from .polyscope_viewer import (
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
