"""Hodge decomposition: split any vector field into curl-free + div-free + harmonic.

Hodge decomposition theorem states any vector field can be written as:
    X = ∇f (curl-free / irrotational) + ∇^⊥g (div-free / incompressible) + h (harmonic)

Algorithmically:
1. Solve Poisson for f: ∇² f = div X  →  f = X_curl_free
2. Solve Poisson for g: ∇² g = curl X  →  g = X_div_free
3. h = X - ∇f - ∇^⊥g
"""
import numpy as np
from typing import Tuple

from ..operators.gradient import gradient
from ..operators.divergence import divergence
from ..operators.curl import curl_vector
from .poisson import solve_poisson

try:
    from scipy.sparse import isspmatrix, csr_matrix
    from scipy.sparse.linalg import spsolve
    _HAS_SCIPY = True
except Exception:
    isspmatrix = lambda x: False
    spsolve = None
    _HAS_SCIPY = False


from ..operators.hodge_star import hodge_star_0, hodge_star_1, hodge_star_2
from ..operators.exterior_derivative import d0, d1

def hodge_decomposition(mesh, v_edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose a 1-form (edge vector field) into curl-free + div-free + harmonic parts.

    Input: v_edges (n_edges,) — values on edges
    Returns: (grad_f, div_free_part, harmonic_part)
             grad_f: curl-free (exact) part
             div_free_part: divergence-free (co-exact) part
             harmonic_part: harmonic residue
    """
    if not _HAS_SCIPY:
        raise ImportError("Hodge decomposition requires a working SciPy installation. "
                          "Your SciPy installation appears to be corrupted or missing.")

    n_e = mesh.n_edges
    n_v = mesh.n_vertices
    n_f = mesh.n_faces

    # Operators
    D0 = d0(mesh)
    D1 = d1(mesh)
    H0 = hodge_star_0(mesh)
    H1 = hodge_star_1(mesh)
    H2 = hodge_star_2(mesh)

    # Step 1: Exact part (curl-free)
    # Solve ∇² f = div(v) => (H0^-1 D0^T H1 D0) f = H0^-1 D0^T H1 v
    # Simplified to: (D0^T H1 D0) f = D0^T H1 v
    L0 = D0.T @ H1 @ D0
    rhs0 = D0.T @ H1 @ v_edges
    
    # Solve with pinning vertex 0 to handle nullspace of constant functions
    # Using the optimized BC logic we implemented earlier
    from scipy.sparse import diags
    mask = np.ones(n_v)
    mask[0] = 0.0
    M = diags(mask)
    L0_fixed = M @ L0 @ M + diags(1.0 - mask)
    rhs0_fixed = M @ rhs0
    f = spsolve(L0_fixed, rhs0_fixed)
    grad_f = D0 @ f

    # Step 2: Co-exact part (div-free)
    # Solve d * d β = d v => (D1 H1^-1 D1^T H2) β = D1 v
    # Let H1_inv be the inverse of diagonal H1
    H1_diag = H1.diagonal()
    H1_inv = diags(1.0 / np.maximum(H1_diag, 1e-12))
    
    L_face = D1 @ H1_inv @ D1.T @ H2
    rhs_face = D1 @ v_edges
    
    # Face Laplacian might have a nullspace if mesh is closed
    # Pin one face to handle it
    mask_f = np.ones(n_f)
    mask_f[0] = 0.0
    Mf = diags(mask_f)
    L_face_fixed = Mf @ L_face @ Mf + diags(1.0 - mask_f)
    rhs_face_fixed = Mf @ rhs_face
    
    beta = spsolve(L_face_fixed, rhs_face_fixed)
    div_free_part = H1_inv @ D1.T @ H2 @ beta

    # Step 3: Harmonic part
    harmonic_part = v_edges - grad_f - div_free_part

    return grad_f, div_free_part, harmonic_part


def hodge_star_decomposition(mesh, v_edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Alternative Hodge decomposition using Hodge star directly.

    Uses: curl-free = ∇(*f) where *f = solution of ∇²(*f) = *div(v)
          div-free = ∇^⊥(*g) where *g = solution of ∇²(*g) = *curl(v)
    """
    # This is mathematically equivalent to hodge_decomposition
    # but uses Hodge star operators more explicitly
    return hodge_decomposition(mesh, v_edges)


def is_divergence_free(mesh, v_edges: np.ndarray, tol: float = 1e-6) -> bool:
    """Check if a 1-form is divergence-free (∇·v = 0 at all vertices)."""
    div_v = divergence(mesh, v_edges)
    return np.max(np.abs(div_v)) < tol


def is_curl_free(mesh, v_edges: np.ndarray, tol: float = 1e-6) -> bool:
    """Check if a 1-form is curl-free (∇×v = 0 at all faces)."""
    curl_v = curl_vector(mesh, v_edges)
    return np.max(np.abs(curl_v)) < tol


__all__ = [
    "hodge_decomposition",
    "hodge_star_decomposition",
    "is_divergence_free",
    "is_curl_free",
]
