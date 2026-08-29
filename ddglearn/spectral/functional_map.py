"""Functional Maps for Shape Correspondence.

Implements the Functional Maps framework (Ovsjanikov et al. 2012) 
for computing correspondences between shapes using spectral embeddings.
"""
import numpy as np


def compute_functional_map(
    mesh1, mesh2, desc1: np.ndarray, desc2: np.ndarray, k: int = 30
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the Functional Map matrix C mapping functions from mesh1 to mesh2.

    Args:
        mesh1: Source HalfEdgeMesh.
        mesh2: Target HalfEdgeMesh.
        desc1: (n_vertices1, n_desc) array of descriptors (e.g., HKS/WKS) for mesh1.
        desc2: (n_vertices2, n_desc) array of descriptors for mesh2.
        k: Number of eigenvectors to use.

    Returns:
        C: (k, k) Functional map matrix.
        evecs1: (n_vertices1, k) eigenfunctions of mesh1.
        evecs2: (n_vertices2, k) eigenfunctions of mesh2.
    """
    from .eigen import eigen_decomposition
    from ..operators.hodge_star import hodge_star_0

    # 1. Compute eigenvalues and eigenvectors
    evals1, evecs1 = eigen_decomposition(mesh1, k=k)
    evals2, evecs2 = eigen_decomposition(mesh2, k=k)

    # 2. Extract Mass matrices for inner product
    M1 = hodge_star_0(mesh1)
    M2 = hodge_star_0(mesh2)

    M1_diag = M1.diagonal() if hasattr(M1, "diagonal") else np.diag(M1)
    M2_diag = M2.diagonal() if hasattr(M2, "diagonal") else np.diag(M2)

    # 3. Project descriptors onto the spectral basis (A = Phi^T * M * Desc)
    A = evecs1.T @ (M1_diag[:, None] * desc1)
    B = evecs2.T @ (M2_diag[:, None] * desc2)

    # 4. Solve for C: min ||C A - B||_F^2
    C = B @ np.linalg.pinv(A)

    return C, evecs1, evecs2


def point_to_point_from_functional_map(C: np.ndarray, evecs1: np.ndarray, evecs2: np.ndarray) -> np.ndarray:
    """Convert a functional map to a point-to-point correspondence using nearest neighbors.
    
    Args:
        C: (k, k) Functional map matrix.
        evecs1: (n_vertices1, k) eigenvectors of source mesh.
        evecs2: (n_vertices2, k) eigenvectors of target mesh.
        
    Returns:
        p2p: (n_vertices1,) array mapping each vertex in mesh1 to a vertex in mesh2.
    """
    # Mapped embeddings: Phi_1 C^T
    mapped_embeddings = evecs1 @ C.T
    
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(evecs2)
        _, p2p = tree.query(mapped_embeddings)
    except ImportError:
        # Fallback O(N^2) if SciPy is missing
        p2p = np.argmin(np.linalg.norm(evecs2[None, :, :] - mapped_embeddings[:, None, :], axis=2), axis=1)
            
    return p2p

__all__ = ["compute_functional_map", "point_to_point_from_functional_map"]