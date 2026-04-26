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
        mesh: HalfEdgeMesh.
        pos: (V, 3) current positions.
        vel: (V, 3) current velocities.
        dt: Time step.
        stiffness: Spring stiffness for edges.
        
    Returns:
        new_pos, new_vel
    """
    n_v = mesh.n_vertices
    forces = np.zeros((n_v, 3))
    
    forces[:, 1] += mass * gravity
    
    from operators.exterior_derivative import edge_list_and_map
    edges, _ = edge_list_and_map(mesh)
    
    for (u, v) in edges:
        p_u, p_v = pos[u], pos[v]
        diff = p_v - p_u
        dist = np.linalg.norm(diff)
        
        orig_pu, orig_pv = mesh.vertices[u], mesh.vertices[v]
        rest_dist = np.linalg.norm(orig_pv - orig_pu)
        
        if dist > 1e-6:
            force = stiffness * (dist - rest_dist) * (diff / dist)
            forces[u] += force
            forces[v] -= force
            
    new_vel = (vel + (forces / mass) * dt) * damping
    if pinned_vertices is not None:
        for p in pinned_vertices:
            new_vel[p] = 0.0
            
    new_pos = pos + new_vel * dt
    return new_pos, new_vel

__all__ = ["cloth_simulation_step"]