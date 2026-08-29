"""Eulerian Fluid Simulation on surfaces using Discrete Exterior Calculus.

Simulates inviscid incompressible fluid flow (Navier-Stokes) directly on the curved manifold
using the vorticity formulation, where vorticity is a 2-form and velocity is a 1-form flux.
"""
import numpy as np

from ..operators.gradient import gradient
from ..operators.hodge_star import hodge_star_1
from .poisson import solve_poisson
from .heat import implicit_heat_step


def fluid_velocity_from_vorticity(mesh, omega: np.ndarray, viscosity: float = 0.0, dt: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """Compute incompressible fluid velocity field from surface vorticity.
    
    In 2D (or on surfaces), incompressible flow is completely determined by a stream function \\psi.
    1. \\Delta \\psi = \\omega
    2. Velocity V = \\nabla^\\perp \\psi = \\star d \\psi
    
    Args:
        mesh: The HalfEdgeMesh instance.
        omega: (n_vertices,) Voriticy distribution (representing the 2-form dual).
        viscosity: Kinematic viscosity for diffusion.
        dt: Timestep.
        
    Returns:
        flux: (n_edges,) 1-form representing velocity flux across edges.
        omega_diffused: (n_vertices,) The diffused vorticity field.
    """
    # 1. Diffuse vorticity if viscous
    if viscosity > 0:
        omega = implicit_heat_step(mesh, omega, t=viscosity * dt, pin_index=None)
        
    # 2. Solve Poisson equation for stream function
    psi = solve_poisson(mesh, omega, pin_index=0, pin_value=0.0)
    
    # 3. Take gradient (d psi -> 1-form)
    grad_psi = gradient(mesh, psi)
    
    # 4. Apply Hodge Star to rotate the gradient by 90 degrees in the tangent plane
    H1 = hodge_star_1(mesh)
    
    if hasattr(H1, "dot"):
        flux = H1.dot(grad_psi)
    else:
        flux = H1 * grad_psi
        
    return flux, omega

__all__ = ["fluid_velocity_from_vorticity"]