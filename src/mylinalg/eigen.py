"""Eigenvalues and eigenvectors related algorithms."""

from typing import Optional
import numpy as np
from mylinalg.decompositions import _householder_reflector
from mylinalg.decompositions import _qr_householder
from mylinalg.utils import Matrix, NPMatrix, check_matrix
from mylinalg.utils import ZERO_TOL, is_zero


def standard_power_iteration(
    A: Matrix,
    shift: Optional[float] = None,
    max_iterations: int = 100,
    convergence_tol: Optional[float] = None,
) -> tuple[float, NPMatrix]:
    A = check_matrix(A)
    m, n = A.shape
    if m != n:
        raise ValueError("Non-square matrix")
    if shift is not None:
        A[:] -= shift * np.identity(m, dtype=A.dtype)
    max_iterations = max(1, max_iterations)
    convergence_tol = ZERO_TOL if convergence_tol is None else convergence_tol
    v0 = np.ones((m, 1), dtype=A.dtype)
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
    if shift is None:
        return lambda1.item(), v1
    return lambda1.item() + shift, v1


def hessenberg_reduction(A: Matrix) -> tuple[NPMatrix, NPMatrix]:
    A = check_matrix(A)
    m, n = A.shape
    if m != n:
        raise ValueError("Non-square matrix")
    S = A.copy()
    H = np.identity(m, dtype=A.dtype)
    for i in range(m - 2):
        H_i = np.identity(m, dtype=A.dtype)
        H_i[i + 1 :, i + 1 :] = _householder_reflector(S[i + 1 :, [i]])
        H_i.dot(S, out=S)
        S.dot(H_i, out=S)
        H.dot(H_i, out=H)
    return H, S


def qr_method(
    A: Matrix,
    max_iterations: int = 100,
    convergence_tol: Optional[float] = None,
) -> tuple[list[float], NPMatrix]:
    A = check_matrix(A)
    m, n = A.shape
    if m != n:
        raise ValueError("Non-square matrix")
    if not is_zero(A - A.T).all():
        raise NotImplementedError("Only symmetric matrices")
    max_iterations = max(1, max_iterations)
    convergence_tol = ZERO_TOL if convergence_tol is None else convergence_tol
    H, A_i = hessenberg_reduction(A)
    iteration = 0
    for i in range(m, 1, -1):
        shift = np.identity(i, dtype=A_i.dtype)
        while iteration < max_iterations:
            np.fill_diagonal(shift, A_i[i - 1, i - 1])
            A_i[:i, :i] -= shift
            Qi, Ri = _qr_householder(A_i[:i, :i])
            H[:, :i].dot(Qi, out=H[:, :i])
            A_i[:i, :i] = Ri.dot(Qi) + shift
            iteration += 1
            if np.abs(A_i[i - 1, i - 2]) < convergence_tol:
                break
        else:
            break
    return np.diagonal(A_i).tolist(), H
