"""Geometry helpers: curvature computations and shape operators."""

from .curvature import (
    gaussian_curvature,
    mean_curvature,
    mean_curvature_vector,
    principal_curvatures,
)
from .shape_operator import (
    compute_vertex_normals,
    shape_operator_tensor,
    principal_directions,
)
from .optimal_transport import sinkhorn_wasserstein
from .parameterization import harmonic_parameterization

__all__ = [
    "gaussian_curvature",
    "mean_curvature",
    "mean_curvature_vector",
    "principal_curvatures",
    "compute_vertex_normals",
    "shape_operator_tensor",
    "principal_directions",
    "sinkhorn_wasserstein",
    "harmonic_parameterization",
]
