"""Package utilities."""

from numbers import Real
from collections.abc import Sequence
from typing import Optional
import numpy as np


type Matrix = Sequence[Real] | Sequence[Sequence[Real]] | np.ndarray


def check_matrix(
    A: Matrix,
    m: Optional[int] = None,
    n: Optional[int] = None,
) -> np.ndarray:
    return
