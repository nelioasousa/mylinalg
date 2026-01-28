"""Package utilities."""

from numbers import Real
from collections.abc import Sequence
from typing import Optional
import numpy as np
from numpy.typing import NDArray


ZERO_TOL = 1e-10


TargetDtype = np.float64
CheckDtype = np.bool
type Matrix = tuple[Sequence[Real], int, int] | Sequence[Sequence[Real]] | np.ndarray
type NPMatrix = NDArray[TargetDtype]
type NPBoolMatrix = NDArray[CheckDtype]


def check_matrix(
    A: Matrix,
    m: Optional[int] = None,
    n: Optional[int] = None,
) -> NDArray:
    if len(A) == 3 and isinstance(A[0], Sequence) and isinstance(A[1], int):
        A, m, n = A
    matrix = np.asarray(A, dtype=TargetDtype)
    if m is None and n is None:
        if len(matrix.shape) < 2:
            # Preference for column vectors
            matrix.resize((matrix.size, 1))
        return matrix
    m = matrix.size // n if m is None else m
    n = matrix.size // m if n is None else n
    matrix.resize((m, n))
    return matrix


def is_zero(A: NPMatrix) -> NPBoolMatrix:
    return np.abs(A) < ZERO_TOL
