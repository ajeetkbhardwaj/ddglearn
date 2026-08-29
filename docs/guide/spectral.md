# Spectral Geometry

Eigenpairs of the Laplacian and spectral shape descriptors.

## Eigen Decomposition

```python
from ddglearn.spectral.eigen import eigen_decomposition

evals, evecs = eigen_decomposition(mesh, k=20)
# evals: (k,) eigenvalues
# evecs: (n_v, k) eigenvectors
```

Uses `scipy.sparse.linalg.eigsh` for sparse or dense `eigh` fallback.

## Heat Kernel Signature (HKS)

```python
from ddglearn.spectral.hks import compute_hks

times, hks = compute_hks(mesh, k=50, times=None)
# times: (T,) time values (auto-selected if None)
# hks:   (n_v, T) signatures
```

$$\text{HKS}(x, t) = \sum_i e^{-\lambda_i t}\, \phi_i(x)^2$$

HKS captures the rate of heat diffusion at each vertex across multiple time scales. It is isometry-invariant and useful for shape matching.

## Wave Kernel Signature (WKS)

```python
from ddglearn.spectral.wks import compute_wks

energies, wks = compute_wks(mesh, k=100, times=None)
# energies: (T,) log-energy values
# wks:      (n_v, T) signatures
```

WKS (Aubry et al. 2011) uses a Gaussian window on the spectral domain, capturing wave propagation at different frequencies. More discriminative than HKS for local features.

## Chebyshev Filters

Chebyshev polynomial approximations for spectral graph convolutions without eigendecomposition:

```python
from ddglearn.spectral.chebyshev import scaled_laplacian, chebyshev_filter

L_sym = scaled_laplacian(mesh)    # scaled to [-1, 1]
filters = chebyshev_filter(mesh, x, order=3)  # [T_0(L)x, T_1(L)x, T_2(L)x, T_3(L)x]
```

Each filter `T_k(L)x` captures frequency content at scale `k`. Useful for building CNN-like features on meshes.

## Functional Maps

Correspondence between two shapes via their spectral representations:

```python
from ddglearn.spectral.functional_map import (
    compute_functional_map,
    point_to_point_from_functional_map,
)

# desc1: (n1, n_desc) descriptor vectors on mesh1
# desc2: (n2, n_desc) descriptor vectors on mesh2
C, evecs1, evecs2 = compute_functional_map(mesh1, mesh2, desc1, desc2, k=30)
# C: (k, k) functional map matrix

correspondence = point_to_point_from_functional_map(C, evecs1, evecs2)
# correspondence: (n1,) vertex indices in mesh2
```

The functional map (Ovsjanikov et al. 2012) encodes shape correspondence as a small matrix `C` in the spectral domain. `point_to_point_from_functional_map` converts this to vertex-to-vertex correspondences via nearest neighbor.
