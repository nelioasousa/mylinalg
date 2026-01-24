"""Package utilities."""

from numbers import Real
from collections.abc import Sequence
from typing import Optional
import numpy as np
from numpy.typing import NDArray


TargetDtype = np.float64
type Matrix = Sequence[Real] | Sequence[Sequence[Real]] | np.ndarray
type NPMatrix = NDArray[TargetDtype]


def check_matrix(
    A: Matrix,
    m: Optional[int] = None,
    n: Optional[int] = None,
) -> NDArray:
    matrix = np.asarray(A, dtype=TargetDtype)
    if m is None and n is None:
        if len(matrix.shape) < 2:
            # Preference for column vectors
            matrix.resize((m.size, 1))
        return matrix
    m = matrix.size // n if m is None else m
    n = matrix.size // m if n is None else n
    matrix.resize((m, n))
    return matrix
