# DDG Framework Examples

This directory contains comprehensive demonstration scripts showcasing the capabilities of the Discrete Differential Geometry (DDG) framework.

## Available Demos

1. **`curvature_demo.py`**
   Visualizes discrete Gaussian curvature, Mean curvature, and principal curvature directions natively on the manifold.

2. **`pde_demo.py`**
   Demonstrates the Heat Method for Geodesic distance computation and implicit time-stepping for heat diffusion PDEs.

3. **`spectral_demo.py`**
   Extracts the Laplace-Beltrami eigen-decomposition and visualizes the Heat Kernel Signature (HKS) and Wave Kernel Signature (WKS) shape descriptors.

4. **`advanced_demo.py`**
   Shows off the cutting-edge features: Sinkhorn Wasserstein optimal transport, Eulerian fluid simulation via Discrete Exterior Calculus, and grid-based mesh decimation.

## Running the examples

For the best experience, install `polyscope` to enable the interactive 3D viewer:
```bash
pip install polyscope
```
