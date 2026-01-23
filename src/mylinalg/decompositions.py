"""Matrix decompositons."""

import numpy as np
from mylinalg.utils import Matrix, check_matrix
from typing import Literal


type Pivoting = Literal["none", "partial", "complete"]


def rr(
    A: Matrix,
    pivoting: Pivoting = "partial",
) -> tuple[Matrix, Matrix, Matrix]:
    return


def rref(
    A: Matrix,
    pivoting: Pivoting = "partial",
) -> tuple[Matrix, Matrix, Matrix]:
    return


def lu(
    A: Matrix,
    pivoting: Pivoting = "none",
) -> tuple[Matrix, Matrix]:
    return


def cholesky(A: Matrix):
    return
