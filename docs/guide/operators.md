# DEC Operators

Discrete Exterior Calculus (DEC) operators on triangle meshes.

## Exterior Derivatives

```python
from ddglearn.operators.exterior_derivative import d0, d1

D0 = d0(mesh)   # (n_edges, n_vertices)  vertices -> edges: -1 at source, +1 at target
D1 = d1(mesh)   # (n_faces, n_edges)     edges -> faces: +1/-1 by orientation
```

## Hodge Stars

```python
from ddglearn.operators.hodge_star import hodge_star_0, hodge_star_1, hodge_star_2

*0 = hodge_star_0(mesh)   # (n_v, n_v) diagonal: Voronoi vertex areas
*1 = hodge_star_1(mesh)   # (n_e, n_e) diagonal: cotangent weights (clamped ≥ 0)
*2 = hodge_star_2(mesh)   # (n_f, n_f) diagonal: face areas
```

The Hodge star \*1 clamps negative cotangent weights to 0, ensuring the resulting Laplacian is PSD.

## Laplacian

```python
from ddglearn.operators.laplacian import laplacian_0

L = laplacian_0(mesh)   # (n_v, n_v) weak Laplacian: L = d0^T *1 d0
```

**Convention:** The weak Laplacian `L = d0^T *1 d0` is symmetric positive semi-definite. The strong Laplacian is `-M^{-1} L`. In PDE solves, always keep the mass matrix on the RHS:

```python
# Correct: L u = M f
# Wrong:   M^{-1} L u = f  (don't use this form)
```

## Gradient

```python
from ddglearn.operators.gradient import gradient, gradient_vector

grad_u = gradient(mesh, u)             # (n_e,) edge gradient of scalar u
grad_v = gradient_vector(mesh, u)      # (n_f, 3) 3D gradient vectors per face
```

## Divergence

```python
from ddglearn.operators.divergence import divergence, divergence_face_vector

div_v = divergence(mesh, v_edges)       # (n_v,) divergence of 1-form v
div_X = divergence_face_vector(mesh, X) # (n_v,) divergence of constant-per-face vector X (n_f, 3)
```

## Curl

```python
from ddglearn.operators.curl import curl_scalar, curl_vector

lap_u = curl_scalar(mesh, u)   # (n_v,) Laplace-Beltrami of scalar u
curl_v = curl_vector(mesh, v)  # (n_f,) curl of 1-form v (edge -> face values)
```

## Connection Laplacian

For tangent vector fields (used in parallel transport and shape operator):

```python
from ddglearn.operators.connection import (
    compute_vertex_bases,       # (n, 2, 3) tangent basis per vertex
    connection_laplacian,       # (2n, 2n) sparse connection Laplacian
    vertex_holonomy,            # (n,) holonomy = angle defect
)

L_conn, bases = connection_laplacian(mesh)
```
