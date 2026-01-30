"""MyLinAlg package."""

import numpy as _np
from mylinalg import decompositions
from mylinalg import eigen
from mylinalg import processing
from mylinalg import solvers

_np.seterr(divide="raise", invalid="raise", over="raise")
