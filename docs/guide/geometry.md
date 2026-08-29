# Geometry

Curvature computation, shape operators, optimal transport, parameterization, and mesh decimation.

## Curvature

```python
from ddglearn.geometry.curvature import (
    gaussian_curvature,       # (n,) angle-deficit per vertex
    mean_curvature_vector,    # (n,3) mean curvature normal via cotangent Laplacian
    mean_curvature,           # (n,) scalar: 0.5 * ||Hn||
    principal_curvatures,     # (k1, k2) from K and H
)

K = gaussian_curvature(mesh)
H = mean_curvature(mesh)
k1, k2 = principal_curvatures(mesh)
```

Gaussian curvature uses the angle-deficit formula:

$$K_v = \frac{2\pi - \sum_{i} \theta_i}{A_v}$$

where $\theta_i$ are face angles at vertex $v$ and $A_v$ is the Voronoi area.

## Shape Operator

```python
from ddglearn.geometry.shape_operator import (
    compute_vertex_normals,   # (n,3) area-weighted normals
    shape_operator_tensor,    # (n,3,3) symmetric 3x3 curvature tensor per vertex
    principal_directions,     # (k1_dirs, k2_dirs) principal curvature directions
)

S = shape_operator_tensor(mesh)       # (n, 3, 3)
k1_dirs, k2_dirs = principal_directions(mesh)  # each (n, 3)
```

The shape operator tensor is computed via edge-based cotangent weights projected to the tangent plane using `np.einsum`.

## Optimal Transport

```python
from ddglearn.geometry.optimal_transport import sinkhorn_wasserstein

dist, u, v = sinkhorn_wasserstein(mesh, p, q, t=1e-3)
```

Convolutional Wasserstein distance (Solomon et al. 2015). Heat-regularized Sinkhorn in O(N). The parameter `t` controls the tradeoff between accuracy and smoothness.

## Parameterization

```python
from ddglearn.geometry.parameterization import harmonic_parameterization

uv = harmonic_parameterization(mesh)  # (n, 2) disk UV mapping
```

Harmonic parameterization maps boundary vertices to a unit circle and solves for interior UV coordinates via harmonic interpolation. Works only for disk-topology meshes (single boundary component).

## Mesh Decimation

```python
from ddglearn.geometry.decimation import decimate_mesh

simplified = decimate_mesh(mesh, grid_resolution=0.05)
```

Grid-based vertex clustering. Larger `grid_resolution` = fewer polygons.
