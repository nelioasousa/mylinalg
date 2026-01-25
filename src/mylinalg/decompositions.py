"""Matrix decompositons."""

import numpy as np
from mylinalg.utils import TargetDtype, NPMatrix, Matrix
from mylinalg.utils import check_matrix, is_zero
from typing import Literal, Optional


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


def _rr_spine(rr: NPMatrix, i: int, j: int):
    rr_step = np.identity(rr.shape[0], dtype=rr.dtype)
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        rr_step[i + 1 :, i] = -1 * rr[i + 1 :, j] / rr[i, j]
    return rr_step


def _ref_none(A: NPMatrix) -> tuple[None, None, NPMatrix, NPMatrix]:
    m, n = A.shape
    # Compute also the LU decomposition
    L = np.identity(m, dtype=A.dtype)
    ref = A.copy()  # `ref` equal the U matrix in the LU decomposition
    j = 0
    for i in range(m):
        while j < n:
            if is_zero(ref[i:, j]).all():
                j += 1
                continue
            rr_step = _rr_spine(ref, i, j)
            rr_step.dot(ref, out=ref)
            rr_step[i + 1 :, i] = -1 * rr_step[i + 1 :, i]
            L.dot(rr_step, out=L)
            j += 1
            break
        else:
            break
    # LU = A
    return L, ref, None, None


def _ref_partial(A: NPMatrix) -> tuple[None, NPMatrix, NPMatrix, NPMatrix]:
    m, n = A.shape
    # Compute also the LU decomposition
    L = np.identity(m, dtype=A.dtype)
    P = np.identity(m, dtype=A.dtype)
    ref = A.copy()  # `ref` equal the U matrix in the LU decomposition
    j = 0
    for i in range(m):
        while j < n:
            if is_zero(ref[i:, j]).all():
                j += 1
                continue
            best_row = np.argmax(np.abs(ref[i:, j])) + i
            if best_row != i:
                P_i = _get_exchange(m, i, best_row)
                P_i.dot(P, out=P)
                P_i.dot(ref, out=ref)
                P_i.dot(L, out=L)
                L.dot(P_i, out=L)
            rr_step = _rr_spine(ref, i, j)
            rr_step.dot(ref, out=ref)
            rr_step[i + 1 :, i] = -1 * rr_step[i + 1 :, i]
            L.dot(rr_step, out=L)
            j += 1
            break
        else:
            break
    # LU = PA
    return L, ref, P, None


def _ref_complete(A: NPMatrix) -> tuple[NPMatrix, NPMatrix, NPMatrix, NPMatrix]:
    m, n = A.shape
    # Compute also the LU decomposition
    L = np.identity(m, dtype=A.dtype)
    P = np.identity(m, dtype=A.dtype)
    Q = np.identity(n, dtype=A.dtype)
    ref = A.copy()  # `ref` equal the U matrix in the LU decomposition
    j = 0
    for i in range(m):
        while j < n:
            if is_zero(ref[i:, j]).all():
                j += 1
                continue
            best_r, best_c = np.unravel_index(
                np.argmax(np.abs(ref[i:, j:])), shape=(m - i, n - j)
            )
            best_r += i
            best_c += j
            if best_r != i:
                P_i = _get_exchange(m, i, best_r)
                P_i.dot(P, out=P)
                P_i.dot(ref, out=ref)
                P_i.dot(L, out=L)
                L.dot(P_i, out=L)
            if best_c != j:
                C_j = _get_exchange(n, j, best_c)
                Q.dot(C_j, out=Q)
                ref.dot(C_j, out=ref)
            rr_step = _rr_spine(ref, i, j)
            rr_step.dot(ref, out=ref)
            rr_step[i + 1 :, i] = -1 * rr_step[i + 1 :, i]
            L.dot(rr_step, out=L)
            j += 1
            break
        else:
            break
    # LU = PAQ
    return L, ref, P, Q


def ref(
    A: Matrix,
    pivoting: Pivoting = "partial",
    return_LU_decomposition: bool = False,
) -> NPMatrix | tuple[NPMatrix, NPMatrix, Optional[NPMatrix], Optional[NPMatrix]]:
    A = check_matrix(A)
    if pivoting == "none":
        res = _ref_none(A)
    elif pivoting == "partial":
        res = _ref_partial(A)
    else:
        res = _ref_complete(A)
    if return_LU_decomposition:
        return res
    return res[1]


def rref(
    A: Matrix,
    pivoting: Pivoting = "partial",
) -> tuple[NPMatrix, NPMatrix] | tuple[NPMatrix, NPMatrix, NPMatrix]:
    return


def _lu_doolittle(A: NPMatrix) -> tuple[NPMatrix, NPMatrix]:
    m, n = A.shape
    L = np.identity(m, dtype=A.dtype)
    U = np.zeros((m, n), dtype=A.dtype)
    for i in range(m):
        U[[i], i:] = A[[i], i:] - L[[i], :i].dot(U[:i, i:])
        with np.errstate(divide="raise", invalid="raise", over="raise"):
            L[i + 1 :, [i]] = (A[i + 1 :, [i]] - L[i + 1 :, :i].dot(U[:i, [i]])) / U[i, i]
    return L, U


def lu(
    A: Matrix,
    pivoting: Pivoting = "none",
) -> tuple[NPMatrix, NPMatrix, Optional[NPMatrix], Optional[NPMatrix]]:
    A = check_matrix(A)
    if pivoting == "none":
        L, U = _lu_doolittle(A)
        return L, U, None, None
    return ref(A, pivoting=pivoting, return_LU_decomposition=True)


def cholesky(A: Matrix) -> tuple[NPMatrix, NPMatrix]:
    return
