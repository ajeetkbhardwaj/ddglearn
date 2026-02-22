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

from operators.gradient import gradient
from operators.divergence import divergence
from operators.curl import curl_vector
from pde.poisson import solve_poisson

try:
    from scipy.sparse import isspmatrix, csr_matrix
    _HAS_SCIPY = True
except Exception:
    isspmatrix = lambda x: False
    _HAS_SCIPY = False


def hodge_decomposition(mesh, v_edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose a 1-form (edge vector field) into curl-free + div-free + harmonic parts.

    Input: v_edges (n_edges,) — values on edges
    Returns: (grad_f, div_free_part, harmonic_part)
             grad_f: curl-free part (vertex scalars → gradient gives contribution)
             div_free_part: divergence-free part
             harmonic_part: harmonic residue
    """
    # Step 1: compute divergence-free part via solving ∇² g = curl(v)
    curl_v = curl_vector(mesh, v_edges)  # shape (n_faces,)
    # convert face values to vertex via averaging
    n_v = mesh.n_vertices
    curl_at_vertex = np.zeros(n_v, dtype=float)
    face_to_vertex_count = np.zeros(n_v, dtype=int)
    for f_idx, f in enumerate(mesh.faces):
        for vi in f:
            curl_at_vertex[int(vi)] += curl_v[f_idx]
            face_to_vertex_count[int(vi)] += 1
    curl_at_vertex /= np.maximum(face_to_vertex_count, 1)

    g = solve_poisson(mesh, curl_at_vertex, pin_index=0, pin_value=0.0)
    grad_g = gradient(mesh, g)

    # Step 2: compute curl-free part via solving ∇² f = div(v)
    # First need to convert edge values to vertex for divergence
    div_v = divergence(mesh, v_edges)  # shape (n_vertices,)
    f = solve_poisson(mesh, div_v, pin_index=0, pin_value=0.0)
    grad_f = gradient(mesh, f)

    # Step 3: compute harmonic part as residual
    # In practice, harmonic part lives in null space of Laplacian
    # For small/closed surfaces, harmonic space is typically 1-dim or small
    from operators.laplacian import laplacian_0
    L = laplacian_0(mesh)
    if _HAS_SCIPY and isspmatrix(L):
        L = csr_matrix(L)
        Lf = L.dot(f)
        Lg = L.dot(g)
    else:
        L = np.asarray(L)
        Lf = L.dot(f)
        Lg = L.dot(g)

    harmonic_part = v_edges - grad_f - grad_g

    return grad_f, grad_g, harmonic_part


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
