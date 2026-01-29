"""Matrix decompositions."""

from typing import Literal, Optional
import numpy as np
from mylinalg.utils import NPMatrix, Matrix
from mylinalg.utils import check_matrix, is_zero, ZERO_TOL


type Pivoting = Literal["none", "partial", "complete"]


def _householder_reflector(A: NPMatrix) -> NPMatrix:
    m, n = A.shape
    if n != 1:
        raise ValueError("`A` must be a column vector.")
    norm = np.linalg.norm(A)
    u = A.copy()
    u[0, 0] += norm if A[0, 0] >= 0 else (-1 * norm)
    u[:] /= np.linalg.norm(u)
    H = np.identity(m, dtype=A.dtype)
    H[:] -= 2 * u.dot(u.T)
    return H


def _qr_householder(A: NPMatrix):
    m, n = A.shape
    Q = np.identity(m, dtype=A.dtype)
    R = A.copy()
    for i in range(min(m - 1, n)):
        H_i = np.identity(m, dtype=A.dtype)
        H_i[i:, i:] = _householder_reflector(R[i:, [i]])
        H_i.dot(R, out=R)
        Q.dot(H_i, out=Q)
    return Q, R


def rank_revealing_qr(
    A: Matrix, independence_tol: Optional[float] = None
) -> tuple[NPMatrix, NPMatrix, NPMatrix]:
    """Retorna uma decomposição QR onde Q é uma base ortonormal de A.

    Entrada:
        `A (Matrix)` - Matriz de entrada a ser decomposta.
        `independence_tol (None | float)` - Tolerância para considerar que uma coluna de `A` colapsou na origem (é combinação linear de outras colunas de `A`).

    Saída:
        `tuple[NPMatrix, NPMatrix, NPMatrix]` - Tripla de matrizes `(P, Q, R)`, de modo que `A @ P = Q @ R`.
    """
    A = check_matrix(A)
    independence_tol = ZERO_TOL if independence_tol is None else independence_tol
    _, n = A.shape
    P = np.identity(n, dtype=A.dtype)
    Q = A.copy()
    R = np.zeros_like(P)
    for j in range(n):
        best_column = np.argmax(np.pow(Q[:, j:], 2).sum(axis=0)) + j
        P[[j, best_column]] = P[[best_column, j]]
        Q[:, [j, best_column]] = Q[:, [best_column, j]]
        R[:, [j, best_column]] = R[:, [best_column, j]]
        R[j, j] = np.linalg.norm(Q[:, j])
        if R[j, j] < independence_tol:
            Q = Q[:, :j]
            R = R[:j, :]
            break
        Q[:, j] /= R[j, j]
        for i in range(j + 1, n):
            R[j, i] = Q[:, j].dot(Q[:, i])
            Q[:, i] -= R[j, i] * Q[:, j]
    return P, Q, R


def _qr_gram_schmidt(A: NPMatrix, independence_tol: float) -> tuple[NPMatrix, NPMatrix]:
    _, n = A.shape
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


def _rr_spine(rr: NPMatrix, i: int, j: int, invert: bool = False):
    rr_step = np.identity(rr.shape[0], dtype=rr.dtype)
    if invert:
        rr_step[:i, i] = -1 * rr[:i, j] / rr[i, j]
    else:
        rr_step[i + 1 :, i] = -1 * rr[i + 1 :, j] / rr[i, j]
    return rr_step


def _lu_gauss_none(
    A: NPMatrix,
    column_lim: Optional[int] = None,
) -> tuple[NPMatrix, NPMatrix, None, None, list[tuple[int, int]]]:
    m, n = A.shape
    column_lim = n if column_lim is None else min(n, column_lim)
    L = np.identity(m, dtype=A.dtype)
    U = A.copy()
    j = 0
    pivots = []
    for i in range(m):
        while j < column_lim:
            if is_zero(U[i:, j]).all():
                j += 1
                continue
            if np.abs(U[i, j]) < ZERO_TOL:
                raise ZeroDivisionError("Zero pivot encountered")
            pivots.append((i, j))
            rr_step = _rr_spine(U, i, j)
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
    column_lim: Optional[int] = None,
) -> tuple[NPMatrix, NPMatrix, NPMatrix, None, list[tuple[int, int]]]:
    m, n = A.shape
    column_lim = n if column_lim is None else min(n, column_lim)
    P = np.identity(m, dtype=A.dtype)
    L = np.identity(m, dtype=A.dtype)
    U = A.copy()
    j = 0
    pivots = []
    for i in range(m):
        while j < column_lim:
            if is_zero(U[i:, j]).all():
                j += 1
                continue
            pivots.append((i, j))
            best_row = np.argmax(np.abs(U[i:, j])) + i
            if best_row != i:
                P_i = np.identity(m, dtype=A.dtype)
                P_i[[i, best_row]] = P_i[[best_row, i]]
                P_i.dot(P, out=P)
                P_i.dot(U, out=U)
                P_i.dot(L, out=L)
                L.dot(P_i, out=L)
            rr_step = _rr_spine(U, i, j)
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
    column_lim: Optional[int] = None,
) -> tuple[NPMatrix, NPMatrix, NPMatrix, NPMatrix, list[tuple[int, int]]]:
    m, n = A.shape
    column_lim = n if column_lim is None else min(n, column_lim)
    P = np.identity(m, dtype=A.dtype)
    Q = np.identity(n, dtype=A.dtype)
    L = np.identity(m, dtype=A.dtype)
    U = A.copy()
    j = 0
    pivots = []
    for i in range(m):
        while j < column_lim:
            if is_zero(U[i:, j]).all():
                j += 1
                continue
            pivots.append((i, j))
            best_r, best_c = np.unravel_index(
                np.argmax(np.abs(U[i:, j:column_lim])),
                shape=(m - i, column_lim - j),
            )
            best_r += i
            best_c += j
            if best_r != i:
                P_i = np.identity(m, dtype=A.dtype)
                P_i[[i, best_r]] = P_i[[best_r, i]]
                P_i.dot(P, out=P)
                P_i.dot(U, out=U)
                P_i.dot(L, out=L)
                L.dot(P_i, out=L)
            if best_c != j:
                C_j = np.identity(n, dtype=A.dtype)
                C_j[[j, best_c]] = C_j[[best_c, j]]
                Q.dot(C_j, out=Q)
                U.dot(C_j, out=U)
            rr_step = _rr_spine(U, i, j)
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
        L[i + 1 :, [i]] = (A[i + 1 :, [i]] - L[i + 1 :, :i].dot(U[:i, [i]])) / U[i, i]
    return L, U


def lu(
    A: Matrix,
    pivoting: Pivoting = "none",
) -> tuple[NPMatrix, NPMatrix, Optional[NPMatrix], Optional[NPMatrix]]:
    """Decomposição LU.

    Entrada:
        `A (Matrix)` - Matriz a ser decomposta.
        `pivoting (Pivoting)` - Tipo de pivotação a ser executada. Pode ser `"none"`, `"partial"` ou `"complete"`.

    Saída:
        `tuple[NPMatrix, NPMatrix, Optional[NPMatrix], Optional[NPMatrix]]` - Quadra de matrizes `(L, U, P, Q)`, de modo que `P @ A @ Q = L @ U`.
    """
    A = check_matrix(A)
    if pivoting == "none":
        L, U = _lu_doolittle(A)
        return L, U, None, None
    if pivoting == "partial":
        L, U, P, Q, _ = _lu_gauss_partial(A)
    else:
        L, U, P, Q, _ = _lu_gauss_complete(A)
    return L, U, P, Q


def qr(A: Matrix) -> tuple[NPMatrix, NPMatrix]:
    """Decomposição QR usando transformações de Householder.

    Entrada:
        `A (Matrix)` - Matriz a ser decomposta.

    Saída:
        `tuple[NPMatrix, NPMatrix]` - Par de matrizes `(Q, R)`, de modo que `A = Q @ R`.
    """
    A = check_matrix(A)
    return _qr_householder(A)


def cholesky(A: Matrix) -> NPMatrix:
    """Decomposição de Cholesky.

    Entrada:
        `A (Matrix)` - Matriz a ser decomposta.

    Saída:
        `NPMatrix` - Matriz `L`, de modo que `A = L @ L.T`.
    """
    A = check_matrix(A)
    n = len(A)
    L = np.zeros_like(A)
    for j in range(n):
        L[j:, [j]] = A[j:, [j]] - L[j:, :j].dot(L[[j], :j].T)
        L[j, j] = np.sqrt(L[j, j])
        L[j + 1 :, j] /= L[j, j]
    return L


def _bidiagonal_reduction(A: NPMatrix) -> tuple[NPMatrix, NPMatrix, NPMatrix]:
    m, n = A.shape
    U = np.identity(m, dtype=A.dtype)
    B = A.copy()
    V_T = np.identity(n, dtype=A.dtype)
    l_lim = min(m - 1, n)
    r_lim = min(m, n - 2)
    for i in range(max(l_lim, r_lim)):
        if i < l_lim:
            U_i = np.identity(m, dtype=A.dtype)
            U_i[i:, i:] = _householder_reflector(B[i:, [i]])
            U.dot(U_i, out=U)
            U_i.dot(B, out=B)
        if i < r_lim:
            V_i = np.identity(n, dtype=A.dtype)
            V_i[i + 1 :, i + 1 :] = _householder_reflector(B[[i], i + 1 :].T)
            V_i.dot(V_T, out=V_T)
            B.dot(V_i, out=B)
    return U, B, V_T


def svd(
    A: Matrix,
    max_iterations: int = 100,
    convergence_tol: Optional[float] = None,
) -> tuple[NPMatrix, NPMatrix, NPMatrix]:
    """Decomposição SVD (Singular Value Decomposition).

    Entrada:
        `A (Matrix)` - Matriz a ser decomposta.

    Saída:
        `tuple[NPMatrix, NPMatrix, NPMatrix]` - Tripla de matrizes `(U, S, V)`, de modo que `A = U @ S @ V`.
    """
    A = check_matrix(A)
    m, n = A.shape
    U_1, B, VT_1 = _bidiagonal_reduction(A)
    if m < n:
        B_t = B.dot(B.T)
    else:
        B_t = B.T.dot(B)
    max_iterations = max(1, max_iterations)
    convergence_tol = ZERO_TOL if convergence_tol is None else convergence_tol
    H = np.identity(B_t.shape[0], dtype=B_t.dtype)
    iteration = 0
    for i in range(B_t.shape[0], 1, -1):
        shift = np.identity(i, dtype=B_t.dtype)
        while iteration < max_iterations:
            np.fill_diagonal(shift, B_t[i - 1, i - 1])
            B_t[:i, :i] -= shift
            Qi, Ri = _qr_householder(B_t[:i, :i])
            H[:, :i] = H[:, :i].dot(Qi)
            B_t[:i, :i] = Ri.dot(Qi) + shift
            iteration += 1
            if np.abs(B_t[i - 1, i - 2]) < convergence_tol:
                break
        else:
            break
    Sigma = np.diag(np.sqrt(np.diagonal(B_t)))
    SigmaInv = np.diag(1 / np.diagonal(Sigma))
    if m < n:
        U = U_1.dot(H)
        V_T = SigmaInv.dot(U.T).dot(A)
    else:
        V_T = H.T.dot(VT_1)
        U = A.dot(V_T.T).dot(SigmaInv)
    return U, Sigma, V_T
