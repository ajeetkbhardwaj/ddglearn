# Tutorial: Shape Fingerprinting with Spectral Descriptors

**Problem:** How do you tell if two 3D shapes are "the same" even when they're bent, stretched, or posed differently? You need a *shape fingerprint* — a descriptor that's invariant to bending.

**Solution:** The Heat Kernel Signature (HKS) and Wave Kernel Signature (WKS) are spectral descriptors that capture the "shape DNA" at each vertex using Laplacian eigenpairs.

## Step 1: Compute the HKS

```python
from ddglearn.core.mesh_io import load_mesh
from ddglearn.core.halfedge import HalfEdgeMesh
from ddglearn.spectral.hks import compute_hks

vertices, faces = load_mesh("data/bunny.obj")
mesh = HalfEdgeMesh(vertices, faces)

times, hks = compute_hks(mesh, k=50)
print(f"HKS shape: {hks.shape}  ({mesh.n_vertices} vertices x {len(times)} time scales)")
```

## Step 2: Visualize HKS at different scales

Each column of the HKS matrix is a spatial function on the surface. Different time scales capture different features:

```python
from ddglearn.core.visualization import save_mesh_snapshot

# Short times → local features (tips, ridges)
save_mesh_snapshot(vertices, faces, hks[:, 0],
    filename="tutorial_hks_short.html",
    title=f"HKS at short time t={times[0]:.4f}")

# Long times → global shape
save_mesh_snapshot(vertices, faces, hks[:, -1],
    filename="tutorial_hks_long.html",
    title=f"HKS at long time t={times[-1]:.2f}")
```

**What you'll see:** Short-time HKS highlights geometric features like ear tips and the nose. Long-time HKS is smooth and captures the overall shape.

## Step 3: Compute the WKS

WKS uses a different spectral window and captures finer details:

```python
from ddglearn.spectral.wks import compute_wks

energies, wks = compute_wks(mesh, k=50)
print(f"WKS shape: {wks.shape}  ({mesh.n_vertices} vertices x {len(energies)} energy levels)")
```

## Step 4: Compare HKS vs WKS

```python
# WKS at fine scale
save_mesh_snapshot(vertices, faces, wks[:, 0],
    filename="tutorial_wks_fine.html",
    title="WKS at fine scale")

# WKS at coarse scale
save_mesh_snapshot(vertices, faces, wks[:, -1],
    filename="tutorial_wks_coarse.html",
    title="WKS at coarse scale")
```

## Step 5: Use HKS for vertex matching

Find the most distinctive vertices — ones whose HKS differs most from the average:

```python
import numpy as np

# Mean HKS across time
mean_hks = hks.mean(axis=1)  # (n_verts,)
std_hks = hks.std(axis=1)    # high std = distinctive

# Most distinctive vertices
top_k = 10
distinctive = np.argsort(std_hks)[-top_k:]
print(f"Most distinctive vertices: {distinctive}")

# Find the "most typical" vertex (closest to mean HKS)
hks_mean_global = hks.mean(axis=0)  # (T,)
dists_to_mean = np.linalg.norm(hks - hks_mean_global, axis=1)
typical = np.argmin(dists_to_mean)
print(f"Most typical vertex: {typical}")
```

## Step 6: Shape signature for retrieval

Collapse HKS into a single shape signature — useful for database retrieval:

```python
# Per-shape signature: mean HKS at each time scale
shape_signature = hks.mean(axis=0)  # (T,)
print(f"Shape signature: {shape_signature[:5]}...")

# Compare two shapes by comparing their signatures
vertices2, faces2 = load_mesh("data/teapot.obj")
mesh2 = HalfEdgeMesh(vertices2, faces2)
times2, hks2 = compute_hks(mesh2, k=50)
sig2 = hks2.mean(axis=0)

# Cosine similarity
from numpy.linalg import norm
cos_sim = np.dot(shape_signature, sig2) / (norm(shape_signature) * norm(sig2))
print(f"Bunny vs Teapot similarity: {cos_sim:.3f}")
```

## Key insight

HKS and WKS are **isometry-invariant** — they don't change when you bend the surface (as long as you don't stretch or tear it). This means a curled-up bunny and a stretched-out bunny have the same spectral fingerprint.

## Complete script

```python
"""Shape fingerprinting tutorial — HKS and WKS spectral descriptors."""
import numpy as np
from ddglearn.core.mesh_io import load_mesh
from ddglearn.core.halfedge import HalfEdgeMesh
from ddglearn.spectral.hks import compute_hks
from ddglearn.spectral.wks import compute_wks
from ddglearn.core.visualization import save_mesh_snapshot

vertices, faces = load_mesh("data/bunny.obj")
mesh = HalfEdgeMesh(vertices, faces)

# HKS
times, hks = compute_hks(mesh, k=50)
save_mesh_snapshot(vertices, faces, hks[:, 0], filename="tutorial_hks_short.html",
    title=f"HKS short t={times[0]:.4f}")
save_mesh_snapshot(vertices, faces, hks[:, -1], filename="tutorial_hks_long.html",
    title=f"HKS long t={times[-1]:.2f}")

# WKS
energies, wks = compute_wks(mesh, k=50)
save_mesh_snapshot(vertices, faces, wks[:, 0], filename="tutorial_wks_fine.html",
    title="WKS fine scale")
save_mesh_snapshot(vertices, faces, wks[:, -1], filename="tutorial_wks_coarse.html",
    title="WKS coarse scale")

# Shape signature
sig_bunny = hks.mean(axis=0)
vertices2, faces2 = load_mesh("data/teapot.obj")
mesh2 = HalfEdgeMesh(vertices2, faces2)
_, hks2 = compute_hks(mesh2, k=50)
sig_teapot = hks2.mean(axis=0)
cos_sim = np.dot(sig_bunny, sig_teapot) / (np.linalg.norm(sig_bunny) * np.linalg.norm(sig_teapot))
print(f"Bunny vs Teapot similarity: {cos_sim:.3f}")
```
