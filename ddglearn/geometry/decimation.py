"""Mesh decimation and simplification algorithms.

Provides robust grid-based vertex clustering to heavily downsample
high-resolution meshes, bypassing the topological fragility of edge-collapse queues.
"""
import numpy as np

def decimate_mesh(mesh, grid_resolution: float = 0.05):
    """Decimate a mesh using spatial grid clustering.
    
    Args:
        mesh: The HalfEdgeMesh to downsample.
        grid_resolution: Size of the spatial clustering grid cells. 
                         Larger values = fewer polygons.
                         
    Returns:
        A new HalfEdgeMesh representing the simplified geometry.
    """
    V = mesh.vertices
    F = mesh.faces
    
    if hasattr(V, "cpu"):
        V = V.cpu().numpy()
        F = F.cpu().numpy()
        
    # 1. Quantize vertices into 3D grid bins
    min_bound = np.min(V, axis=0)
    coords = np.floor((V - min_bound) / grid_resolution).astype(np.int32)
    
    # 2. Find unique cells and remap
    unique_coords, inverse_indices, counts = np.unique(
        coords, axis=0, return_inverse=True, return_counts=True
    )
    
    # 3. Create new vertices as the centroid of the collapsed cell
    n_new = len(unique_coords)
    new_V = np.zeros((n_new, 3), dtype=V.dtype)
    np.add.at(new_V, inverse_indices, V)
    new_V /= counts[:, None]
    
    # 4. Remap faces to new vertex indices
    new_F = inverse_indices[F]
    
    # 5. Remove degenerate faces (faces where vertices collapsed to same cell)
    valid = (new_F[:, 0] != new_F[:, 1]) & \
            (new_F[:, 1] != new_F[:, 2]) & \
            (new_F[:, 2] != new_F[:, 0])
            
    new_F = new_F[valid]
    
    # 6. Clean up unreferenced vertices
    used_vertices = np.unique(new_F)
    if len(used_vertices) < n_new:
        remap = np.full(n_new, -1, dtype=np.int32)
        remap[used_vertices] = np.arange(len(used_vertices))
        new_V = new_V[used_vertices]
        new_F = remap[new_F]
        
    from ..core.halfedge import HalfEdgeMesh
    
    # We disable manifold validation because severe grid clustering 
    # can naturally produce non-manifold pinch points.
    return HalfEdgeMesh(new_V, new_F, validate_manifold=False)

__all__ = ["decimate_mesh"]