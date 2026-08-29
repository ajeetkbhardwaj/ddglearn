# Tutorial: Shape Correspondence via Functional Maps

**Problem:** Given two different 3D shapes (e.g., a bunny in two poses), find which point on one corresponds to which point on the other. This is fundamental for shape matching, texture transfer, and statistical shape analysis.

**Solution:** Functional Maps (Ovsjanikov et al. 2012) compute correspondences in the spectral domain — a small matrix that maps functions from one shape to another.

## Step 1: Load two shapes

We'll use the teapot (with boundary) for this demo since it has distinct features:

```python
from ddglearn.core.mesh_io import load_mesh
from ddglearn.core.halfedge import HalfEdgeMesh

v1, f1 = load_mesh("data/bunny.obj")
v2, f2 = load_mesh("data/teapot.obj")
mesh1 = HalfEdgeMesh(v1, f1)
mesh2 = HalfEdgeMesh(v2, f2)
print(f"Mesh 1: {mesh1.n_vertices}V  Mesh 2: {mesh2.n_vertices}V")
```

## Step 2: Compute spectral descriptors

HKS and WKS serve as "descriptor functions" on each shape — they encode local geometry in a way that's consistent across shapes:

```python
from ddglearn.spectral.hks import compute_hks
from ddglearn.spectral.wks import compute_wks

_, hks1 = compute_hks(mesh1, k=50)
_, hks2 = compute_hks(mesh2, k=50)

# Stack HKS and WKS as descriptor channels
from ddglearn.spectral.wks import compute_wks
_, wks1 = compute_wks(mesh1, k=50)
_, wks2 = compute_wks(mesh2, k=50)

desc1 = np.hstack([hks1, wks1])  # (n1, 100)
desc2 = np.hstack([hks2, wks2])  # (n2, 100)
print(f"Descriptors: {desc1.shape} and {desc2.shape}")
```

## Step 3: Compute the functional map

```python
import numpy as np
from ddglearn.spectral.functional_map import (
    compute_functional_map,
    point_to_point_from_functional_map,
)

C, evecs1, evecs2 = compute_functional_map(mesh1, mesh2, desc1, desc2, k=30)
print(f"Functional map C: {C.shape}")  # (30, 30)
```

The functional map `C` is a small `k x k` matrix. To map a function `f` from mesh1 to mesh2: `f_mapped = C @ f_spectral`.

## Step 4: Convert to point-to-point correspondence

```python
correspondence = point_to_point_from_functional_map(C, evecs1, evecs2)
print(f"Correspondence shape: {correspondence.shape}")  # (n1,)
print(f"Each vertex in mesh1 maps to a vertex in mesh2")
```

## Step 5: Visualize the correspondence

Transfer a color function from mesh1 to mesh2:

```python
from ddglearn.core.visualization import save_mesh_snapshot

# Color mesh1 by x-coordinate (horizontal position)
color_func = v1[:, 0]  # x-coordinate as descriptor

# Transfer to mesh2 via correspondence
transferred = color_func[correspondence]

save_mesh_snapshot(v2, f2, transferred,
    filename="tutorial_correspondence_transfer.html",
    title="X-coordinate transferred from bunny to teapot")
```

## Step 6: Evaluate map quality

Check how well the functional map preserves descriptor compatibility:

```python
# Descriptor compatibility: C A ≈ B
from ddglearn.operators.hodge_star import hodge_star_0

M1 = hodge_star_0(mesh1)
M2 = hodge_star_0(mesh2)
M1d = M1.diagonal() if hasattr(M1, "diagonal") else np.diag(M1)
M2d = M2.diagonal() if hasattr(M2, "diagonal") else np.diag(M2)

A = evecs1.T @ (M1d[:, None] * desc1[:, :1])  # first descriptor
B = evecs2.T @ (M2d[:, None] * desc2[:, :1])

B_approx = C @ A
error = np.linalg.norm(B - B_approx) / (np.linalg.norm(B) + 1e-12)
print(f"Relative descriptor error: {error:.4f}")
```

A small error means the functional map successfully captures the geometric relationship between the shapes.

## How it works

1. **Spectral embedding:** Each shape gets a basis of Laplacian eigenvectors $\phi_1, \ldots, \phi_k$
2. **Descriptor projection:** Descriptors (HKS/WKS) are projected onto this basis
3. **Linear solve:** Find $C$ such that $C \cdot A \approx B$ where $A$, $B$ are projected descriptors
4. **Nearest neighbor:** Convert spectral map to point-to-point via KD-tree

## Complete script

```python
"""Shape correspondence tutorial — functional maps between shapes."""
import numpy as np
from ddglearn.core.mesh_io import load_mesh
from ddglearn.core.halfedge import HalfEdgeMesh
from ddglearn.spectral.hks import compute_hks
from ddglearn.spectral.wks import compute_wks
from ddglearn.spectral.functional_map import (
    compute_functional_map,
    point_to_point_from_functional_map,
)
from ddglearn.core.visualization import save_mesh_snapshot

v1, f1 = load_mesh("data/bunny.obj")
v2, f2 = load_mesh("data/teapot.obj")
mesh1 = HalfEdgeMesh(v1, f1)
mesh2 = HalfEdgeMesh(v2, f2)

# Descriptors
_, hks1 = compute_hks(mesh1, k=50)
_, hks2 = compute_hks(mesh2, k=50)
_, wks1 = compute_wks(mesh1, k=50)
_, wks2 = compute_wks(mesh2, k=50)
desc1 = np.hstack([hks1, wks1])
desc2 = np.hstack([hks2, wks2])

# Functional map
C, evecs1, evecs2 = compute_functional_map(mesh1, mesh2, desc1, desc2, k=30)
correspondence = point_to_point_from_functional_map(C, evecs1, evecs2)

# Transfer x-coordinate
color_func = v1[:, 0]
transferred = color_func[correspondence]
save_mesh_snapshot(v2, f2, transferred,
    filename="tutorial_correspondence_transfer.html",
    title="Bunny x-coord transferred to teapot")
```
