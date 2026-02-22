"""Demo: compute and visualize Gaussian and mean curvature on a small mesh.

This script uses matplotlib to plot vertex-based Gaussian curvature as
colors and the mean curvature vector as quivers (projected to XY).
"""
import sys
import pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.halfedge import HalfEdgeMesh
from geometry.curvature import gaussian_curvature, mean_curvature_vector

import matplotlib.pyplot as plt


def visualize_curvature(mesh, title="Curvature"):
    K = gaussian_curvature(mesh)
    Hn = mean_curvature_vector(mesh)

    V = mesh.vertices
    xy = V[:, :2]

    fig, ax = plt.subplots(figsize=(5, 5))
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=K, cmap="viridis", s=80, edgecolor='k')
    plt.colorbar(sc, ax=ax, label="Gaussian curvature")

    # show mean curvature vector projected to XY plane
    scale = 0.1 * max(1.0, np.max(np.linalg.norm(Hn, axis=1)))
    ax.quiver(xy[:, 0], xy[:, 1], Hn[:, 0], Hn[:, 1], angles='xy', scale_units='xy', scale=1.0/scale, color='r')

    ax.set_title(title)
    ax.set_aspect('equal')
    plt.show()


def main():
    # simple pyramid mesh (square base + apex) to show nontrivial curvature
    V = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, 0.5],
    ])
    F = np.array([
        [0, 1, 4],
        [1, 2, 4],
        [2, 3, 4],
        [3, 0, 4],
        [0, 1, 2],
        [0, 2, 3],
    ])
    from core.mesh_io import load_obj
    path = '../data/bunny.obj'
    V, F = load_obj(path)
    mesh = HalfEdgeMesh(V, F)
    visualize_curvature(mesh, title="Gaussian & Mean Curvature (pyramid)")


if __name__ == '__main__':
    main()
