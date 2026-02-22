"""Small demo: compute eigenpairs and HKS on a tiny mesh."""
import sys
import pathlib
import numpy as np

# Ensure project root (parent of this examples folder) is on sys.path so
# imports like `core` and `spectral` resolve when running this script.
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.halfedge import HalfEdgeMesh
from spectral.eigen import eigen_decomposition
from spectral.hks import compute_hks


def main():
    V = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    F = np.array([[0, 1, 2]])
    m = HalfEdgeMesh(V, F)

    vals, vecs = eigen_decomposition(m, k=3)
    print('eigenvalues:', vals)

    times, hks = compute_hks(m, k=3)
    print('HKS shape', hks.shape)


if __name__ == '__main__':
    main()
