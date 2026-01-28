"""Linear equations solvers."""

import numpy as np
from mylinalg.processing import rref
from mylinalg.decompositions import lu
from mylinalg.utils import NPMatrix, Matrix, check_matrix


def matrix_inverse(A: Matrix) -> NPMatrix:
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
    return


def forward_substitution(A: Matrix, b: Matrix) -> NPMatrix:
    return


def gauss_jordan_solver(A: Matrix, b: Matrix) -> NPMatrix:
    return


def lu_solver(A: Matrix, b: Matrix) -> NPMatrix:
    return


def least_squares_solver(A: Matrix, b: Matrix) -> NPMatrix:
    return
