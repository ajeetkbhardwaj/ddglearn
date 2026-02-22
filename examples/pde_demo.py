"""Demo: solve Poisson and run a heat step on a small mesh."""
import sys
import pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.halfedge import HalfEdgeMesh
from pde.poisson import solve_poisson
from pde.heat import implicit_heat


def main():
    V = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5,0.5,0.5]])
    F = np.array([[0,1,3],[1,2,3],[2,0,3]])
    mesh = HalfEdgeMesh(V, F)

    n = mesh.n_vertices
    f = np.zeros(n)
    # unit source at apex
    f[3] = 1.0

    u = solve_poisson(mesh, f)
    print('Poisson solution u:', u)

    u_heat = implicit_heat(mesh, u, t=0.01, steps=5)
    print('Heat-evolved u (5 steps):', u_heat)


if __name__ == '__main__':
    main()
