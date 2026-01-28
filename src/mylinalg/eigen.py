"""Eigenvalues and eigenvectors related algorithms."""

from typing import Optional
import numpy as np
from mylinalg.utils import Matrix, NPMatrix, check_matrix
from mylinalg.utils import ZERO_TOL


def standard_power_iteration(
    A: Matrix,
    max_iterations: int = 100,
    convergence_tol: Optional[float] = None,
) -> tuple[float, NPMatrix]:
    A = check_matrix(A)
    max_iterations = max(1, max_iterations)
    convergence_tol = ZERO_TOL if convergence_tol is None else convergence_tol
    v0 = np.ones((A.shape[0], 1), dtype=A.dtype)
    v1 = np.empty_like(v0)
    lambda0 = 0.0
    for _ in range(max_iterations):
        A.dot(v0, out=v1)
        lambda1 = v0.T.dot(v1) / v0.T.dot(v0)
        v1[:] /= np.linalg.norm(v1)
        if np.abs(lambda1 - lambda0) < convergence_tol:
            break
        v0[:] = v1
        lambda0 = lambda1
    return lambda1.item(), v1
