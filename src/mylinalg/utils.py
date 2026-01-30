"""Package utilities."""

from numbers import Real
from collections.abc import Sequence
import numpy as np
from numpy.typing import NDArray


ZERO_TOL = 1e-10
ZERO_TOL_ITER = 1e-20


TargetDtype = np.float64
CheckDtype = np.bool
type Matrix = Sequence[Real] | Sequence[Sequence[Real]] | np.ndarray
type NPMatrix = NDArray[TargetDtype]
type NPBoolMatrix = NDArray[CheckDtype]


def check_matrix(A: Matrix) -> NPMatrix:
    """Converte a matriz `A` para uma matriz NumPy com a presição float64.

    Entrada:
        `A (Matrix)` - Matriz a ser convertida.

    Saída:
        `NPMatrix` - Matriz NumPy de 2 dimensões com precisão float64.
    """
    matrix = np.asarray(A, dtype=TargetDtype)
    num_dims = len(matrix.shape)
    if num_dims > 2:
        raise ValueError("`A` must be a 2-dimensional matrix")
    if num_dims != 2:
        # Preference for column vectors
        matrix = matrix.reshape((matrix.size, 1))
    return matrix


def is_zero(A: NPMatrix) -> NPBoolMatrix:
    """Checa se os alementos de `A` são zeros segundo uma tolerância padrão.

    Entrada:
        `A (NPMatrix)` - Matriz a ser checada.

    Saída:
        `NPBoolMatrix` - Matriz booleana com as mesmas dimensões de `A`, com `True` onde o elemento é considerado efetivamente zero.
    """
    return np.abs(A) < ZERO_TOL
