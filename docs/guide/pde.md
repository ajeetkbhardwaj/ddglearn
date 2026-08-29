# PDE Solvers

Partial differential equation solvers on triangle meshes.

## Poisson Solver

Solve $\Delta u = f$ with boundary conditions:

```python
from ddglearn.pde.poisson import solve_poisson

u = solve_poisson(mesh, f,
    pin_index=0, pin_value=0.0,
    dirichlet_indices=None,
    dirichlet_values=None,
    neumann_indices=None,
    neumann_values=None,
)
```

Internally solves the weak form $L_c\, u = *0\, f$ where $L_c = d_0^T\,{*1}\,d_0$.

**Boundary conditions:**

| Type | How to specify |
|---|---|
| Pinned | `pin_index`, `pin_value` — fix a single vertex |
| Dirichlet | `dirichlet_indices`, `dirichlet_values` — fix multiple vertices |
| Neumann | `neumann_indices`, `neumann_values` — natural BC (default = 0) |

## Heat Diffusion

```python
from ddglearn.pde.heat import implicit_heat_step, implicit_heat

u_next = implicit_heat_step(mesh, u, t=1e-5)
u_final = implicit_heat(mesh, u0, t=1e-5, steps=100)
```

Implicit Euler: solves $(H_0 + t\,W)\,u_{next} = H_0\,u$ where $H_0 = {*0}$ and $W = d_0^T\,{*1}\,d_0$.

The time step $t$ controls diffusion scale — use small values ($10^{-5}$ to $10^{-3}$) for local features.

## Wave Equation

```python
from ddglearn.pde.wave import wave_step, simulate_wave

u_next = wave_step(mesh, u_prev, u_curr, dt=0.01)
frames = simulate_wave(mesh, u0, dt=0.01, steps=200)
```

Crank-Nicolson / Newmark-beta implicit scheme. **Unconditionally stable** for any `dt`, second-order in time. Solves:

$$(H_0 + \tfrac{dt^2}{2}\,W)\,u^{n+1} = 2\,H_0\,u^n - (H_0 + \tfrac{dt^2}{2}\,W)\,u^{n-1}$$

## Geodesic Distance (Heat Method)

```python
from ddglearn.pde.geodesics import geodesic_distance, heat_method_geodesics

d = geodesic_distance(mesh, source_indices=[0], time_scale=1e-3)
# Methods: "poisson" (default), "varadhan", "fmm_graph"

# Batch: all-pairs or nearest-source
distances = heat_method_geodesics(mesh, source_indices=[0, 100, 200], batch=False)
nearest = heat_method_geodesics(mesh, source_indices=[0, 100, 200], batch=True)
```

The Heat Method (Crane et al. 2013):

1. Solve heat equation: $(I + t\,L)\,u = \delta_{source}$
2. Evaluate vector field: $X = -\nabla u / \|\nabla u\|$
3. Solve Poisson: $L\,\phi = \nabla \cdot X$

Geodesic distance: $\phi(x) - \min \phi$.

## Hodge Decomposition

Decompose a vector field into curl-free + divergence-free + harmonic components:

```python
from ddglearn.pde.hodge_decomposition import hodge_decomposition

grad_f, div_free, harmonic = hodge_decomposition(mesh, v_edges)
# grad_f:     curl-free (gradient) component
# div_free:   divergence-free (curl) component
# harmonic:   harmonic (both curl-free and div-free)

is_div_free = is_divergence_free(mesh, v_edges)
is_curl_free = is_curl_free(mesh, v_edges)
```

## Fluids

Incompressible Euler flow on surfaces:

```python
from ddglearn.pde.fluids import fluid_velocity_from_vorticity

flux, omega = fluid_velocity_from_vorticity(mesh, omega,
    viscosity=0.0, dt=0.01)
```

Converts vorticity to velocity via stream function Poisson solve, then takes gradient. Returns edge flux `(n_e,)` and diffused vorticity `(n_v,)`.

## Cloth Simulation

Mass-spring cloth dynamics:

```python
from ddglearn.pde.cloth import cloth_simulation_step

new_pos, new_vel = cloth_simulation_step(
    mesh, pos, vel,
    dt=0.01, mass=1.0, stiffness=1000.0, damping=0.99,
    gravity=-9.81, pinned_vertices=[0, 1, 2],
)
```

Semi-implicit Euler with spring forces along edges, gravity, and damping. Pin vertices by index.
