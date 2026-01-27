"""Matrix decompositons."""

import numpy as np
from mylinalg.utils import TargetDtype, NPMatrix, Matrix
from mylinalg.utils import check_matrix, is_zero, ZERO_TOL
from typing import Literal, Optional


type Pivoting = Literal["none", "partial", "complete"]


def rank_revealing_qr(
    A: Matrix, independence_tol: Optional[float] = None
) -> tuple[NPMatrix, NPMatrix, NPMatrix]:
    A = check_matrix(A)
    independence_tol = ZERO_TOL if independence_tol is None else independence_tol
    n = A.shape[1]
    P = np.identity(n, dtype=A.dtype)
    Q = A.copy()
    R = np.zeros_like(P)
    for j in range(n):
        best_column = np.argmax(np.pow(Q[:, j:], 2).sum(axis=0)) + j
        P[[j, best_column]] = P[[best_column, j]]
        Q[:, [j, best_column]] = Q[:, [best_column, j]]
        R[j, j] = np.linalg.norm(Q[:, j])
        if R[j, j] >= independence_tol:
            Q[:, j] /= R[j, j]
        for i in range(j + 1, n):
            R[j, i] = Q[:, j].dot(Q[:, i])
            Q[:, i] -= R[j, i] * Q[:, j]
    return P, Q, R


def _qr_gram_schmidt(A: NPMatrix, independence_tol: float) -> tuple[NPMatrix, NPMatrix]:
    n = A.shape[1]
    Q = A.copy()
    R = np.zeros((n, n), dtype=A.dtype)
    for j in range(n):
        R[j, j] = np.linalg.norm(Q[:, j])
        if R[j, j] >= independence_tol:
            Q[:, j] /= R[j, j]
        for i in range(j + 1, n):
            R[j, i] = Q[:, j].dot(Q[:, i])
            Q[:, i] -= R[j, i] * Q[:, j]
    return Q, R


def _get_exchange(n: int, i: int, j: int) -> NPMatrix:
    exchange = np.identity(n, dtype=TargetDtype)
    exchange[[i, j]] = exchange[[j, i]]
    return exchange


def _rr_spine(rr: NPMatrix, i: int, j: int, invert: bool = False):
    rr_step = np.identity(rr.shape[0], dtype=rr.dtype)
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        if invert:
            rr_step[:i, i] = -1 * rr[:i, j] / rr[i, j]
        else:
            rr_step[i + 1 :, i] = -1 * rr[i + 1 :, j] / rr[i, j]
    return rr_step


def _lu_gauss_none(
    A: NPMatrix,
) -> tuple[NPMatrix, NPMatrix, None, None, list[tuple[int, int]]]:
    m, n = A.shape
    L = np.identity(m, dtype=A.dtype)
    U = A.copy()
    j = 0
    pivots = []
    for i in range(m):
        while j < n:
            if is_zero(U[i:, j]).all():
                j += 1
                continue
            rr_step = _rr_spine(U, i, j)
            pivots.append((i, j))
            rr_step.dot(U, out=U)
            rr_step[i + 1 :, i] = -1 * rr_step[i + 1 :, i]
            L.dot(rr_step, out=L)
            j += 1
            break
        else:
            break
    # LU = A
    return L, U, None, None, pivots


def _lu_gauss_partial(
    A: NPMatrix,
) -> tuple[NPMatrix, NPMatrix, NPMatrix, None, list[tuple[int, int]]]:
    m, n = A.shape
    L = np.identity(m, dtype=A.dtype)
    P = np.identity(m, dtype=A.dtype)
    U = A.copy()
    j = 0
    pivots = []
    for i in range(m):
        while j < n:
            if is_zero(U[i:, j]).all():
                j += 1
                continue
            best_row = np.argmax(np.abs(U[i:, j])) + i
            if best_row != i:
                P_i = _get_exchange(m, i, best_row)
                P_i.dot(P, out=P)
                P_i.dot(U, out=U)
                P_i.dot(L, out=L)
                L.dot(P_i, out=L)
            rr_step = _rr_spine(U, i, j)
            pivots.append((i, j))
            rr_step.dot(U, out=U)
            rr_step[i + 1 :, i] = -1 * rr_step[i + 1 :, i]
            L.dot(rr_step, out=L)
            j += 1
            break
        else:
            break
    # LU = PA
    return L, U, P, None, pivots


def _lu_gauss_complete(
    A: NPMatrix,
) -> tuple[NPMatrix, NPMatrix, NPMatrix, NPMatrix, list[tuple[int, int]]]:
    m, n = A.shape
    L = np.identity(m, dtype=A.dtype)
    P = np.identity(m, dtype=A.dtype)
    Q = np.identity(n, dtype=A.dtype)
    U = A.copy()
    j = 0
    pivots = []
    for i in range(m):
        while j < n:
            if is_zero(U[i:, j]).all():
                j += 1
                continue
            best_r, best_c = np.unravel_index(
                np.argmax(np.abs(U[i:, j:])), shape=(m - i, n - j)
            )
            best_r += i
            best_c += j
            if best_r != i:
                P_i = _get_exchange(m, i, best_r)
                P_i.dot(P, out=P)
                P_i.dot(U, out=U)
                P_i.dot(L, out=L)
                L.dot(P_i, out=L)
            if best_c != j:
                C_j = _get_exchange(n, j, best_c)
                Q.dot(C_j, out=Q)
                U.dot(C_j, out=U)
            rr_step = _rr_spine(U, i, j)
            pivots.append((i, j))
            rr_step.dot(U, out=U)
            rr_step[i + 1 :, i] = -1 * rr_step[i + 1 :, i]
            L.dot(rr_step, out=L)
            j += 1
            break
        else:
            break
    # LU = PAQ
    return L, U, P, Q, pivots


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
    if pivoting == "partial":
        L, U, P, Q, _ = _lu_gauss_partial(A)
    else:
        L, U, P, Q, _ = _lu_gauss_complete(A)
    return L, U, P, Q


def cholesky(A: Matrix) -> NPMatrix:
    A = check_matrix(A)
    n = len(A)
    L = np.zeros_like(A)
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        for j in range(n):
            L[j:, [j]] = A[j:, [j]] - L[j:, :j].dot(L[[j], :j].T)
            L[j, j] = np.sqrt(L[j, j])
            L[j + 1 :, j] /= L[j, j]
    return L
