"""General matrix processing methods."""

from typing import Optional
import numpy as np
from mylinalg.utils import NPMatrix, Matrix, check_matrix, ZERO_TOL
from mylinalg.decompositions import _qr_gram_schmidt, rank_revealing_qr
from mylinalg.decompositions import _lu_gauss_none, _lu_gauss_partial, _lu_gauss_complete
from mylinalg.decompositions import _rr_spine, _householder_reflector
from mylinalg.decompositions import Pivoting


def ref(
    A: Matrix,
    pivoting: Pivoting = "partial",
    return_pivots_loc: bool = False,
    column_lim: Optional[int] = None,
) -> NPMatrix | tuple[NPMatrix, list[tuple[int, int]]]:
    A = check_matrix(A)
    if pivoting == "none":
        lu_dec = _lu_gauss_none(A, column_lim=column_lim)
    elif pivoting == "partial":
        lu_dec = _lu_gauss_partial(A, column_lim=column_lim)
    else:
        lu_dec = _lu_gauss_complete(A, column_lim=column_lim)
    _, ref, *_, pivots = lu_dec
    if return_pivots_loc:
        return ref, pivots
    return ref


def rref(
    A: Matrix,
    pivoting: Pivoting = "partial",
    return_pivots_loc: bool = False,
    column_lim: Optional[int] = None,
) -> NPMatrix | tuple[NPMatrix, list[tuple[int, int]]]:
    A = check_matrix(A)
    if pivoting == "none":
        lu_dec = _lu_gauss_none(A, column_lim=column_lim)
    elif pivoting == "partial":
        lu_dec = _lu_gauss_partial(A, column_lim=column_lim)
    else:
        lu_dec = _lu_gauss_complete(A, column_lim=column_lim)
    _, rref, *_, pivots = lu_dec
    for i, j in pivots:
        rr_step = _rr_spine(rref, i, j, invert=True)
        rr_step[i, i] = 1 / rref[i, j]
        rr_step.dot(rref, out=rref)
    if return_pivots_loc:
        return rref, pivots
    return rref


def gram_schmidt(
    A: Matrix, drop_columns: bool = False, independence_tol: Optional[float] = None
) -> NPMatrix:
    A = check_matrix(A)
    independence_tol = ZERO_TOL if independence_tol is None else independence_tol
    Q, R = _qr_gram_schmidt(A, independence_tol=independence_tol)
    if drop_columns:
        return Q[:, np.diagonal(R) >= independence_tol]
    return Q


def column_space_projector(
    A: Matrix, independence_tol: Optional[float] = None
) -> NPMatrix:
    independence_tol = ZERO_TOL if independence_tol is None else independence_tol
    _, Q, _ = rank_revealing_qr(A, independence_tol=independence_tol)
    if not Q.size:
        raise ValueError("All columns are collapsed onto the zero vector")
    return Q.dot(Q.T)


def householder_reflector(A: Matrix) -> NPMatrix:
    A = check_matrix(A)
    return _householder_reflector(A)
