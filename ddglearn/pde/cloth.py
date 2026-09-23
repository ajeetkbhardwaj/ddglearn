"""Mass-Spring Cloth Simulation.

Simulates cloth dynamics on the mesh manifold using a semi-implicit 
mass-spring system along the edge network.
"""
import numpy as np

def cloth_simulation_step(
    mesh, 
    pos: np.ndarray, 
    vel: np.ndarray, 
    dt: float = 0.01, 
    mass: float = 1.0, 
    stiffness: float = 1000.0, 
    damping: float = 0.99, 
    gravity: float = -9.81, 
    pinned_vertices: list = None
) -> tuple[np.ndarray, np.ndarray]:
    """Perform one step of cloth simulation.
    
    Args:
        mesh (HalfEdgeMesh): The mesh.
        pos (np.ndarray): (V, 3) current positions.
        vel (np.ndarray): (V, 3) current velocities.
        dt (float): Time step.
        stiffness (float): Spring stiffness for edges.

    Returns:
        tuple[np.ndarray, np.ndarray]: (new_pos, new_vel).
    """
    n_v = mesh.n_vertices
    forces = np.zeros((n_v, 3))
    
    forces[:, 1] += mass * gravity
    
    from ..operators.exterior_derivative import edge_list_and_map
    edges, _ = edge_list_and_map(mesh)
    edges_np = np.array(edges, dtype=int)

    u_idx = edges_np[:, 0]
    v_idx = edges_np[:, 1]

    p_u = pos[u_idx]  # (E, 3)
    p_v = pos[v_idx]
    diff = p_v - p_u  # (E, 3)
    dist = np.linalg.norm(diff, axis=1)  # (E,)

    orig_pu = mesh.vertices[u_idx]
    orig_pv = mesh.vertices[v_idx]
    rest_dist = np.linalg.norm(orig_pv - orig_pu, axis=1)  # (E,)

    safe_dist = np.maximum(dist, 1e-6)
    force_vec = stiffness * (dist - rest_dist)[:, None] * (diff / safe_dist[:, None])  # (E, 3)

    np.add.at(forces, u_idx, force_vec)
    np.add.at(forces, v_idx, -force_vec)
            
    new_vel = (vel + (forces / mass) * dt) * damping
    if pinned_vertices is not None:
        for p in pinned_vertices:
            new_vel[p] = 0.0
            
    new_pos = pos + new_vel * dt
    return new_pos, new_vel

__all__ = ["cloth_simulation_step"]