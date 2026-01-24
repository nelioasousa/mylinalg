"""Matrix decompositons."""

import numpy as np
from mylinalg.utils import TargetDtype, NPMatrix, Matrix, check_matrix
from typing import Literal


type Pivoting = Literal["none", "partial", "complete"]


def _get_exchange(n: int, i: int, j: int) -> NPMatrix:
    exchange = np.identity(n, dtype=TargetDtype)
    exchange[[i, j]] = exchange[[j, i]]
    return exchange


# def _get_combination(
#     m: int,
#     n: int,
#     i: int,
#     j: int,
#     i_coef: float,
#     j_coef: float,
#     target: int,
# ) -> NPMatrix:
#     comb = np.eye(m, n, k=0, dtype=TargetDtype)
#     comb[target] = comb[i] * i_coef + comb[j] * j_coef
#     return comb


def _rr_column(rr: NPMatrix, i: int):
    rr_step = np.identity(rr.shape[0], dtype=rr.dtype)
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        rr_step[i + 1 :, i] = -1 * rr[i + 1 :, i] / rr[i, i]
    return rr_step


def _rr_none(A: NPMatrix) -> tuple[NPMatrix, NPMatrix]:
    m, n = A.shape
    acc_steps = np.identity(m, dtype=A.dtype)
    rr = A.copy()
    for i in range(min(m, n)):
        rr_step = _rr_column(rr, i)
        rr_step.dot(rr, out=rr)
        rr_step.dot(acc_steps, out=acc_steps)
    return acc_steps, rr


def _rr_partial(A: NPMatrix) -> tuple[NPMatrix, NPMatrix]:
    m, n = A.shape
    acc_steps = np.identity(m, dtype=A.dtype)
    rr = A.copy()
    for i in range(min(m, n)):
        best_row = np.argmax(rr[i:, i]) + i
        if best_row != i:
            exchange = _get_exchange(m, i, best_row)
            exchange.dot(acc_steps, out=acc_steps)
            exchange.dot(rr, out=rr)
        rr_step = _rr_column(rr, i)
        rr_step.dot(rr, out=rr)
        rr_step.dot(acc_steps, out=acc_steps)
    return acc_steps, rr


def _rr_complete(A: NPMatrix) -> tuple[NPMatrix, NPMatrix, NPMatrix]:
    m, n = A.shape
    acc_lft_steps = np.identity(m, dtype=A.dtype)
    acc_rgt_steps = np.identity(n, dtype=A.dtype)
    rr = A.copy()
    for i in range(min(m, n)):
        best_r, best_c = np.unravel_index(
            np.argmax(np.abs(A[i:, i:])), shape=(m - i, n - i)
        )
        best_r += i
        best_c += i
        if best_r != i:
            exchange = _get_exchange(m, i, best_r)
            exchange.dot(acc_lft_steps, out=acc_lft_steps)
            exchange.dot(rr, out=rr)
        if best_c != i:
            exchange = _get_exchange(n, i, best_c)
            acc_rgt_steps.dot(exchange, out=acc_rgt_steps)
            rr.dot(exchange, out=rr)
        rr_step = _rr_column(rr, i)
        rr_step.dot(rr, out=rr)
        rr_step.dot(acc_lft_steps, out=acc_lft_steps)
    return acc_lft_steps, rr, acc_rgt_steps


def rr(
    A: Matrix,
    pivoting: Pivoting = "partial",
) -> tuple[NPMatrix, NPMatrix] | tuple[NPMatrix, NPMatrix, NPMatrix]:
    A = check_matrix(A)
    if pivoting == "none":
        res = _rr_none(A)
    elif pivoting == "partial":
        res = _rr_partial(A)
    else:
        res = _rr_complete(A)
    return res


def rref(
    A: Matrix,
    pivoting: Pivoting = "partial",
) -> tuple[NPMatrix, NPMatrix] | tuple[NPMatrix, NPMatrix, NPMatrix]:
    return


def lu(
    A: Matrix,
    pivoting: Pivoting = "none",
) -> tuple[NPMatrix, NPMatrix]:
    return


def cholesky(A: Matrix) -> tuple[NPMatrix, NPMatrix]:
    return
