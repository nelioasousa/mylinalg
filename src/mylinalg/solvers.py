"""Linear equations solvers."""

import numpy as np
from mylinalg.processing import rref
from mylinalg.decompositions import lu
from mylinalg.utils import NPMatrix, Matrix, check_matrix, ZERO_TOL


def matrix_inverse(A: Matrix) -> NPMatrix:
    """Computa a inversa da matriz A usando Gauss-Jordan.

    Entrada:
        `A (Matrix)` - Matriz para qual computar a inversa.

    Saída:
        `NPMatrix` - Inversa da matriz `A`.
    """
    A = check_matrix(A)
    m, n = A.shape
    if m != n:
        raise ValueError("Non-square matrix")
    A_aug = np.concatenate((A, np.identity(m, dtype=A.dtype)), axis=1)
    rr, pivots = rref(A_aug, pivoting="partial", return_pivots_loc=True, column_lim=n)
    if len(pivots) != m:
        raise ValueError("Singular matrix")
    return rr[:, m:]


def backward_substitution(A: Matrix, b: Matrix) -> NPMatrix:
    A = check_matrix(A)
    b = check_matrix(b)
    m, n = A.shape
    if m != n:
        raise ValueError("Non-square matrix")
    if b.shape[0] != m:
        raise ValueError("Shape mismatch between `A` and `b`")
    solutions = np.zeros((n, b.shape[1]), dtype=A.dtype)
    for i in range(m - 1, -1, -1):
        if np.abs(A[i, i]) < ZERO_TOL:
            continue
        solutions[i, :] = (b[i, :] - A[[i], i + 1 :].dot(solutions[i + 1 :, :])) / A[i, i]
    return solutions


def forward_substitution(A: Matrix, b: Matrix) -> NPMatrix:
    A = check_matrix(A)
    b = check_matrix(b)
    m, n = A.shape
    if m != n:
        raise ValueError("Non-square matrix")
    if b.shape[0] != m:
        raise ValueError("Shape mismatch between `A` and `b`")
    solutions = np.zeros((n, b.shape[1]), dtype=A.dtype)
    for i in range(m):
        if np.abs(A[i, i]) < ZERO_TOL:
            continue
        solutions[i, :] = (b[i, :] - A[[i], :i].dot(solutions[:i, :])) / A[i, i]
    return solutions


def gauss_jordan_solver(A: Matrix, b: Matrix) -> NPMatrix:
    A = check_matrix(A)
    b = check_matrix(b)
    m, n = A.shape
    if m != n:
        raise ValueError("Non-square matrix")
    if b.shape[0] != m:
        raise ValueError("Shape mismatch between `A` and `b`")
    A_aug = np.concatenate((A, b), axis=1)
    rr, pivots = rref(A_aug, pivoting="partial", return_pivots_loc=True, column_lim=n)
    if len(pivots) != m:
        raise ValueError("Singular matrix")
    return rr[:, m:]


def lu_solver(A: Matrix, b: Matrix) -> NPMatrix:
    return


def least_squares_solver(A: Matrix, b: Matrix) -> NPMatrix:
    return
