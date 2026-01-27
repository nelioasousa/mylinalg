"""General matrix processing methods."""

import numpy as np
from mylinalg.utils import NPMatrix, Matrix, check_matrix
from mylinalg.decompositions import _qr_gram_schmidt


def gram_schmidt(A: Matrix) -> NPMatrix:
    A = check_matrix(A)
    Q, _ = _qr_gram_schmidt(A)
    return Q


def get_least_squares_projector(A: Matrix) -> Matrix:
    return
