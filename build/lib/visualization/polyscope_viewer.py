"""Interactive 3D visualization using Polyscope (if available).

Polyscope is a powerful viewer for point clouds, surface meshes, and scalar/vector fields.
Website: https://polyscope.run/

Install: pip install polyscope
"""

import numpy as np
from typing import Dict, Any

try:
    import polyscope as ps

    _HAS_POLYSCOPE = True
except ImportError:
    _HAS_POLYSCOPE = False
    ps = None


def initialize_viewer(mesh) -> None:
    """Initialize Polyscope with the mesh."""
    if not _HAS_POLYSCOPE:
        print("Polyscope not available. Install with: pip install polyscope")
        return

    ps.init()
    mesh_ps = ps.register_surface_mesh(
        "mesh", mesh.vertices, mesh.faces, enabled=True, transparency=0.3
    )


def plot_scalar_field(mesh, field: np.ndarray, name: str = "scalar_field") -> None:
    """Display scalar field on mesh vertices.

    field: array of shape (n_vertices,) or (n_faces,)
    name: label for the field
    """
    if not _HAS_POLYSCOPE:
        print("Polyscope not available.")
        return

    ps.init()
    mesh_ps = ps.register_surface_mesh(
        "mesh", mesh.vertices, mesh.faces, enabled=True, transparency=0.3
    )

    if len(field) == mesh.n_vertices:
        mesh_ps.add_vertex_scalar_quantity(name, field, enabled=True, cmap="viridis")
    elif len(field) == mesh.n_faces:
        mesh_ps.add_face_scalar_quantity(name, field, enabled=True, cmap="viridis")

    ps.show()


def plot_vector_field(
    mesh, vectors: np.ndarray, name: str = "vector_field", location: str = "vertex"
) -> None:
    """Display vector field on mesh.

    vectors: array of shape (n_vertices, 3) or (n_faces, 3)
    location: "vertex" or "face"
    name: label for the field
    """
    if not _HAS_POLYSCOPE:
        print("Polyscope not available.")
        return

    ps.init()
    mesh_ps = ps.register_surface_mesh(
        "mesh", mesh.vertices, mesh.faces, enabled=True, transparency=0.3
    )

    if location == "vertex":
        mesh_ps.add_vertex_vector_quantity(name, vectors, enabled=True)
    elif location == "face":
        mesh_ps.add_face_vector_quantity(name, vectors, enabled=True)

    ps.show()


def plot_curvature(mesh, scalar_field: str = "gaussian") -> None:
    """Visualize curvature on mesh.

    scalar_field: "gaussian" or "mean"
    """
    from geometry.curvature import gaussian_curvature, mean_curvature

    if scalar_field == "gaussian":
        K = gaussian_curvature(mesh)
        plot_scalar_field(mesh, K, name="Gaussian Curvature")
    elif scalar_field == "mean":
        H = mean_curvature(mesh)
        plot_scalar_field(mesh, H, name="Mean Curvature")
    else:
        print(f"Unknown curvature type: {scalar_field}")


def plot_normals(mesh, scale: float = 0.1) -> None:
    """Visualize surface normals as vector field."""
    from geometry.shape_operator import compute_vertex_normals

    normals = compute_vertex_normals(mesh)
    plot_vector_field(mesh, normals * scale, name="Surface Normals", location="vertex")


def plot_principal_directions(mesh, k: int = 1) -> None:
    """Visualize principal curvature directions.

    k: 1 for k1 directions, 2 for k2 directions
    """
    from geometry.shape_operator import principal_directions

    k1_dirs, k2_dirs = principal_directions(mesh)
    dirs = k1_dirs if k == 1 else k2_dirs
    plot_vector_field(mesh, dirs, name=f"Principal Direction k{k}", location="vertex")


def plot_spectral_modes(mesh, mode: int = 0, k: int = 10) -> None:
    """Visualize a Laplacian eigenvector as scalar field."""
    from spectral.eigen import eigen_decomposition

    vals, vecs = eigen_decomposition(mesh, k=k)
    mode_idx = min(mode, vecs.shape[1] - 1)
    eigenvector = vecs[:, mode_idx]
    eigenvalue = vals[mode_idx]

    plot_scalar_field(
        mesh, eigenvector, name=f"Eigenmode {mode_idx} (λ={eigenvalue:.4f})"
    )


def plot_hks(mesh, time_idx: int = 0, k: int = 20) -> None:
    """Visualize Heat Kernel Signature at a specific time scale."""
    from spectral.hks import compute_hks

    times, hks = compute_hks(mesh, k=k)
    time_idx = min(time_idx, hks.shape[1] - 1)
    plot_scalar_field(mesh, hks[:, time_idx], name=f"HKS (t={times[time_idx]:.4f})")


def plot_geodesic_distance(mesh, source_vertex: int = 0) -> None:
    """Visualize geodesic distance from a source vertex."""
    from pde.geodesics import geodesic_distance

    distances = geodesic_distance(mesh, source_vertex)
    plot_scalar_field(mesh, distances, name=f"Geodesic Distance from v{source_vertex}")


def batch_visualization(mesh, visualizations: Dict[str, Any]) -> None:
    """Create multiple visualizations in one Polyscope session.

    visualizations: dict of {name: (type, data, kwargs)}
        where type is "scalar", "vector", or "mesh"
    """
    if not _HAS_POLYSCOPE:
        print("Polyscope not available.")
        return

    ps.init()
    mesh_ps = ps.register_surface_mesh(
        "mesh", mesh.vertices, mesh.faces, enabled=True, transparency=0.3
    )

    for name, (vis_type, data, kwargs) in visualizations.items():
        if vis_type == "scalar":
            if len(data) == mesh.n_vertices:
                mesh_ps.add_vertex_scalar_quantity(name, data, **kwargs)
            elif len(data) == mesh.n_faces:
                mesh_ps.add_face_scalar_quantity(name, data, **kwargs)
        elif vis_type == "vector":
            location = kwargs.pop("location", "vertex")
            if location == "vertex":
                mesh_ps.add_vertex_vector_quantity(name, data, **kwargs)
            elif location == "face":
                mesh_ps.add_face_vector_quantity(name, data, **kwargs)

    ps.show()


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
