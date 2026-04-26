"""Animation & Interactive PDE Demonstration."""

import sys
import numpy as np
from pathlib import Path

# Ensure the parent directory is in the path so 'import ddg' resolves correctly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ddg import HalfEdgeMesh, load_mesh, implicit_heat_step
from ddg.visualization.polyscope_viewer import animate_field

def get_demo_mesh():
    data_path = Path(__file__).resolve().parents[1] / "data" / "bunny.obj"
    if data_path.exists():
        V, F = load_mesh(str(data_path))
        return HalfEdgeMesh(V, F)
    else:
        V = np.array([[1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]], dtype=float)
        F = np.array([[0,4,2], [2,4,1], [1,4,3], [3,4,0], [0,2,5], [2,1,5], [1,3,5], [3,0,5]], dtype=int)
        return HalfEdgeMesh(V, F)

def main():
    mesh = get_demo_mesh()
    
    print("Setting up Heat Diffusion Animation...")
    u_state = np.zeros(mesh.n_vertices)
    u_state[0] = 100.0  # Hot spot
    
    def update_heat(current_u):
        return implicit_heat_step(mesh, current_u, t=0.005)
        
    print("Launching Interactive Polyscope Animation...")
    print("-> Click 'Run Simulation' or 'Step' in the UI window!")
    
    # We pass **kwargs (vminmax) to lock the scale so it doesn't wash out!
    animate_field(mesh, u_state, update_heat, name="Heat Diffusion", cmap="inferno", vminmax=(0.0, 50.0))

if __name__ == "__main__":
    main()