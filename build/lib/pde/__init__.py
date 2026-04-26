"""PDE solvers: Poisson, heat equation, wave equation, geodesics, Hodge decomposition."""

from .poisson import solve_poisson
from .heat import implicit_heat, implicit_heat_step
from .wave import simulate_wave, wave_step
from .geodesics import geodesic_distance, heat_method_geodesics, solve_heat_diffusion
from .hodge_decomposition import (
    hodge_decomposition,
    hodge_star_decomposition,
    is_divergence_free,
    is_curl_free,
)
from .cloth import cloth_simulation_step

__all__ = [
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
    "cloth_simulation_step",
]
