"""Linear equations solvers."""

import numpy as np
from mylinalg.decompositions import rref, lu
from mylinalg.utils import NPMatrix, Matrix, check_matrix


def backward_substitution(A: Matrix, b: Matrix) -> NPMatrix:
    return


def forward_substitution(A: Matrix, b: Matrix) -> NPMatrix:
    return


def gauss_jordan_solver(
    A: Matrix,
    b: Matrix,
) -> NPMatrix:
    return


def lu_solver(
    A: Matrix,
    b: Matrix,
) -> NPMatrix:
    return
