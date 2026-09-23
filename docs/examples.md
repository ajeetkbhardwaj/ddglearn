# Examples

All examples are in `examples/`. Run any with:

```bash
conda activate ddg
PYTHONPATH=. python examples/01_mesh_basics.py
```

## Example List

| #  | File                              | Topic                                            |
| -- | --------------------------------- | ------------------------------------------------ |
| 01 | `01_mesh_basics.py`             | Half-edge mesh, neighbor traversal, vertex areas |
| 02 | `02_curvature.py`               | Gaussian, mean, principal curvatures             |
| 03 | `03_dec_operators.py`           | d0, d1 incidence matrices                        |
| 04 | `04_hodge_laplacian.py`         | Hodge stars, weak Laplacian                      |
| 05 | `05_gradient_divergence.py`     | Gradient, divergence, Hodge decomposition        |
| 06 | `06_connection.py`              | Connection Laplacian, parallel transport         |
| 07 | `07_shape_operator.py`          | Shape operator tensor, principal directions      |
| 08 | `08_pde_poisson_heat.py`        | Poisson solve, heat diffusion                    |
| 09 | `09_pde_wave.py`                | Wave equation simulation                         |
| 10 | `10_pde_geodesics.py`           | Heat Method geodesic distance                    |
| 11 | `11_pde_cloth_fluids.py`        | Cloth dynamics, surface fluids                   |
| 12 | `12_spectral_hks_wks.py`        | HKS, WKS, spectral descriptors                   |
| 13 | `13_spectral_functional_map.py` | Functional map shape correspondence              |
| 14 | `14_benchmarks.py`              | Performance comparison (vectorized vs naive)     |

## Visualization Output

Examples 08–13 save interactive HTML files to `examples/_output/`:

```bash
open examples/_output/08_heat_diffusion.html
open examples/_output/09_wave.html
```

Each HTML file contains a WebGL-rendered mesh with a time slider and play/pause controls.
