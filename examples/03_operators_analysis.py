"""
Example 03 — Differential Operators on Scalar / Vector Fields
==============================================================
Gradient, divergence, curl, and the cotangent-weight Laplacian.

Modules used:
    ddglearn.operators.gradient             — gradient, gradient_vector
    ddglearn.operators.divergence           — divergence, divergence_face_vector
    ddglearn.operators.curl                 — curl_scalar, curl_vector
    ddglearn.operators.cotangent_laplacian  — cotangent_laplacian
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from ddglearn.core.halfedge import HalfEdgeMesh
from ddglearn.core.mesh_io import load_mesh
from ddglearn.operators.gradient import gradient, gradient_vector
from ddglearn.operators.divergence import divergence, divergence_face_vector
from ddglearn.operators.curl import curl_scalar, curl_vector
from ddglearn.operators.cotangent_laplacian import cotangent_laplacian

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
vertices, faces = load_mesh(os.path.join(DATA, "teapot.obj"))
mesh = HalfEdgeMesh(vertices, faces)

# ---------------------------------------------------------------------------
# 1. Scalar field: height function u = z-coordinate
# ---------------------------------------------------------------------------
u = vertices[:, 2].copy()
print(f"Scalar field u (z-coord): min={u.min():.4f}, max={u.max():.4f}")

# Gradient: vertex scalar → edge 1-form
grad_u = gradient(mesh, u)
print(f"\ngrad(u): shape={grad_u.shape}, |grad|_max={np.abs(grad_u).max():.4f}")

# Gradient vector: vertex scalar → face vectors
grad_v = gradient_vector(mesh, u)
print(f"grad_vector(u): shape={grad_v.shape}")

# ---------------------------------------------------------------------------
# 2. Cotangent Laplacian
# ---------------------------------------------------------------------------
L_cot = cotangent_laplacian(mesh)
Lu = L_cot @ u
print(f"\nCotangent Laplacian: shape={L_cot.shape}")
print(f"L(u) (height laplacian): min={Lu.min():.4f}, max={Lu.max():.4f}")

# Symmetry
diff = L_cot - L_cot.T
print(f"L_cot symmetric? max|diff|={abs(diff).max():.2e}")

# ---------------------------------------------------------------------------
# 3. Divergence: edge 1-form → vertex scalar
# ---------------------------------------------------------------------------
from ddglearn.operators.laplacian import laplacian_0
from ddglearn.operators.hodge_star import hodge_star_0

div_grad = divergence(mesh, grad_u)
print(f"\ndiv(grad(u)): shape={div_grad.shape}, min={div_grad.min():.4f}, max={div_grad.max():.4f}")

# DEC identity: div(grad(u)) = -M^{-1} L_c u  (Strong form)
L_weak = laplacian_0(mesh)
H0 = hodge_star_0(mesh)
M_diag = H0.diagonal() if hasattr(H0, 'diagonal') else np.diag(H0)
M_inv = np.diag(1.0 / np.maximum(M_diag, 1e-30))
L_weak_dense = L_weak.toarray() if hasattr(L_weak, 'toarray') else np.asarray(L_weak)
L_strong_u = M_inv @ L_weak_dense @ u
dec_diff = np.abs(div_grad + L_strong_u).max()
print(f"div(grad(u)) vs -M^-1 L_c u: max diff = {dec_diff:.2e}  (DEC identity)")

# Cotangent Laplacian is a different discretization — close but not identical
print(f"div(grad(u)) vs L_cot(u):    max diff = {np.abs(div_grad - Lu).max():.4f}  (different discretization)")

# ---------------------------------------------------------------------------
# 4. Curl: scalar and vector forms
# ---------------------------------------------------------------------------
curl_s = curl_scalar(mesh, u)
print(f"\ncurl_scalar(u): shape={curl_s.shape}")

curl_v = curl_vector(mesh, grad_u)
print(f"curl_vector(grad_u): shape={curl_v.shape}")

# ---------------------------------------------------------------------------
# 5. Face-vector divergence
# ---------------------------------------------------------------------------
X = grad_v.copy()  # use gradient of height as face vector field
div_X = divergence_face_vector(mesh, X)
print(f"\ndiv_face_vector(X): shape={div_X.shape}, min={div_X.min():.4f}, max={div_X.max():.4f}")

print("\nDone.")
