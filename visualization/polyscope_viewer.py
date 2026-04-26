"""Interactive 3D visualization using Polyscope (if available).

Polyscope is a powerful viewer for point clouds, surface meshes, and scalar/vector fields.
Website: https://polyscope.run/

Install: pip install polyscope
"""

import numpy as np
from typing import Dict, Any, Callable

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
    ps.set_ground_plane_mode("shadow_only")
    mesh_ps = ps.register_surface_mesh(
        "mesh", mesh.vertices, mesh.faces, enabled=True, smooth_shade=True, material="clay"
    )


def plot_scalar_field(mesh, field: np.ndarray, name: str = "scalar_field", cmap: str = "viridis", **kwargs) -> None:
    """Display scalar field on mesh vertices.

    field: array of shape (n_vertices,) or (n_faces,)
    name: label for the field
    cmap: colormap name
    """
    if not _HAS_POLYSCOPE:
        print("Polyscope not available.")
        return

    ps.init()
    ps.set_ground_plane_mode("shadow_only")
    mesh_ps = ps.register_surface_mesh(
        "mesh", mesh.vertices, mesh.faces, enabled=True, smooth_shade=True, material="clay"
    )

    if len(field) == mesh.n_vertices:
        mesh_ps.add_scalar_quantity(name, field, defined_on="vertices", enabled=True, cmap=cmap, **kwargs)
    elif len(field) == mesh.n_faces:
        mesh_ps.add_scalar_quantity(name, field, defined_on="faces", enabled=True, cmap=cmap, **kwargs)

    ps.show()


def plot_vector_field(
    mesh, vectors: np.ndarray, name: str = "vector_field", location: str = "vertex", **kwargs
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
    ps.set_ground_plane_mode("shadow_only")
    mesh_ps = ps.register_surface_mesh(
        "mesh", mesh.vertices, mesh.faces, enabled=True, smooth_shade=True, material="clay"
    )

    if location == "vertex":
        mesh_ps.add_vector_quantity(name, vectors, defined_on="vertices", enabled=True, **kwargs)
    elif location == "face":
        mesh_ps.add_vector_quantity(name, vectors, defined_on="faces", enabled=True, **kwargs)

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
    ps.set_ground_plane_mode("shadow_only")
    mesh_ps = ps.register_surface_mesh(
        "mesh", mesh.vertices, mesh.faces, enabled=True, smooth_shade=True, material="clay"
    )

    for name, (vis_type, data, kwargs) in visualizations.items():
        if vis_type == "scalar":
            if len(data) == mesh.n_vertices:
                mesh_ps.add_scalar_quantity(name, data, defined_on="vertices", **kwargs)
            elif len(data) == mesh.n_faces:
                mesh_ps.add_scalar_quantity(name, data, defined_on="faces", **kwargs)
        elif vis_type == "vector":
            location = kwargs.pop("location", "vertex")
            if location == "vertex":
                mesh_ps.add_vector_quantity(name, data, defined_on="vertices", **kwargs)
            elif location == "face":
                mesh_ps.add_vector_quantity(name, data, defined_on="faces", **kwargs)

    ps.show()


def animate_field(
    mesh, 
    initial_field: np.ndarray, 
    update_fn: Callable[[np.ndarray], np.ndarray], 
    name: str = "animated_field", 
    is_vector: bool = False, 
    cmap: str = "viridis", 
    **kwargs
) -> None:
    """Run an interactive animation of a time-dependent field using a callback function.
    
    update_fn: A function that takes the current state array and returns the next state array.
    """
    if not _HAS_POLYSCOPE:
        print("Polyscope not available. Install with: pip install polyscope")
        return

    import polyscope.imgui as psim

    ps.init()
    ps.set_ground_plane_mode("shadow_only")
    mesh_ps = ps.register_surface_mesh(
        "mesh", mesh.vertices, mesh.faces, enabled=True, smooth_shade=True, material="clay"
    )

    state = {"data": initial_field, "running": False, "step": 0}

    def update_vis():
        if is_vector:
            loc = kwargs.pop("location", "vertex")
            defined_on = "vertices" if loc == "vertex" else "faces"
            mesh_ps.add_vector_quantity(name, state["data"], defined_on=defined_on, enabled=True, **kwargs)
        else:
            defined_on = "vertices" if len(state["data"]) == mesh.n_vertices else "faces"
            mesh_ps.add_scalar_quantity(name, state["data"], defined_on=defined_on, enabled=True, cmap=cmap, **kwargs)

    update_vis()

    def callback():
        changed, state["running"] = psim.Checkbox("Run Simulation", state["running"])
        if psim.Button("Step") or state["running"]:
            state["data"] = update_fn(state["data"])
            state["step"] += 1
            update_vis()
        psim.Text(f"Step: {state['step']}")

    ps.set_user_callback(callback)
    ps.show()
    ps.clear_user_callback()

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
    "animate_field",
]
