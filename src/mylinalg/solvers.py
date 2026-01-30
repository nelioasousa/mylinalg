"""Linear equations solvers."""

import numpy as np
from mylinalg.processing import rref
from mylinalg.decompositions import Pivoting, lu, rank_revealing_qr
from mylinalg.decompositions import _lu_gauss_none, _lu_gauss_partial, _lu_gauss_complete
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


def gauss_elimination_solver(A: Matrix, b: Matrix, pivoting: Pivoting) -> NPMatrix:
    A = check_matrix(A)
    b = check_matrix(b)
    m, n = A.shape
    if m != n:
        raise ValueError("Non-square matrix")
    if b.shape[0] != m:
        raise ValueError("Shape mismatch between `A` and `b`")
    if pivoting == "complete":
        acc_steps, _, U, _, Q, _ = _lu_gauss_complete(A)
        b_mod = acc_steps.dot(b)
        x_mod = backward_substitution(U, b_mod)
        return Q.dot(x_mod)
    if pivoting == "none":
        acc_steps, _, U, *_ = _lu_gauss_none(A)
    else:
        acc_steps, _, U, *_ = _lu_gauss_partial(A)
    b_mod = acc_steps.dot(b)
    return backward_substitution(U, b_mod)


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
    A = check_matrix(A)
    b = check_matrix(b)
    m, n = A.shape
    if m != n:
        raise ValueError("Non-square matrix")
    if b.shape[0] != m:
        raise ValueError("Shape mismatch between `A` and `b`")
    L, U, P, _ = lu(A, pivoting="partial")
    b_mod = P.dot(b)
    y = forward_substitution(L, b_mod)
    return backward_substitution(U, y)


def least_squares_solver(A: Matrix, b: Matrix) -> NPMatrix:
    A = check_matrix(A)
    b = check_matrix(b)
    m, _ = A.shape
    if b.shape[0] != m:
        raise ValueError("Shape mismatch between `A` and `b`")
    P, Q, R = rank_revealing_qr(A)
    R_inv = matrix_inverse(R)
    return P.dot(R_inv).dot(Q.T).dot(b)
