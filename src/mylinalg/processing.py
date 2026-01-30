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
    """Row Echelon Form (REF).

    Processa a matriz `A` para deixá-la em REF.

    Entrada:
        `A (Matrix)` - Matriz a ser processada.
        `pivoting (Pivoting)` - Tipo de pivotação a ser executada. Pode ser `"none"`, `"partial"` ou `"complete"`.
        `return_pivots_loc (bool)` - Quando `True` retorna também as coordenadas dos pivôs.
        `column_lim (None | int)` - Limita até qual coluna proceder com a redução REF. Útil para aplicar o método de Gauss-Jordan para solução de sistemas e inversão de matrizes.

    Saída:
        `NPMatrix | tuple[NPMatrix, list[tuple[int, int]]]` - Matriz `A` em REF, podendo vir acompanhada de uma lista com as coordenadas dos pivôs caso `return_pivots_loc` seja `True`.
    """
    A = check_matrix(A)
    if pivoting == "none":
        lu_dec = _lu_gauss_none(A, column_lim=column_lim)
    elif pivoting == "partial":
        lu_dec = _lu_gauss_partial(A, column_lim=column_lim)
    else:
        lu_dec = _lu_gauss_complete(A, column_lim=column_lim)
    _, _, ref, *_, pivots = lu_dec
    if return_pivots_loc:
        return ref, pivots
    return ref


def rref(
    A: Matrix,
    pivoting: Pivoting = "partial",
    return_pivots_loc: bool = False,
    column_lim: Optional[int] = None,
) -> NPMatrix | tuple[NPMatrix, list[tuple[int, int]]]:
    """Reduced Row Echelon Form (RREF).

    Processa a matriz `A` para deixá-la em RREF.

    Entrada:
        `A (Matrix)` - Matriz a ser processada.
        `pivoting (Pivoting)` - Tipo de pivotação a ser executada. Pode ser `"none"`, `"partial"` ou `"complete"`.
        `return_pivots_loc (bool)` - Quando `True` retorna também as coordenadas dos pivôs.
        `column_lim (None | int)` - Limita até qual coluna proceder com a redução RREF. Útil para aplicar o método de Gauss-Jordan para solução de sistemas e inversão de matrizes.

    Saída:
        `NPMatrix | tuple[NPMatrix, list[tuple[int, int]]]` - Matriz `A` em RREF, podendo vir acompanhada de uma lista com as coordenadas dos pivôs caso `return_pivots_loc` seja `True`.
    """
    A = check_matrix(A)
    if pivoting == "none":
        lu_dec = _lu_gauss_none(A, column_lim=column_lim)
    elif pivoting == "partial":
        lu_dec = _lu_gauss_partial(A, column_lim=column_lim)
    else:
        lu_dec = _lu_gauss_complete(A, column_lim=column_lim)
    _, _, rref, *_, pivots = lu_dec
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
    """Ortogonalização de Gram-Schmidt.

    Entrada:
        `A (Matrix)` - Matriz cujas colunas serão ortogonalizadas e normalizadas.
        `drop_columns (bool)` - Se `True`, dropa colunas que são linearmente dependentes.
        `independence_tol (None | float)` - Tolerância para considerar que uma coluna de `A` colapsou na origem (é combinação de outras colunas de `A`).

    Saída:
        `NPMatrix` - Matriz `A` com as colunas ortogonalizadas e normalizadas. Se `drop_columns` for `True`, pode haver menos colunas que a matriz original `A`, pois colunas que foram consideradas linearmente dependentes foram eliminadas.
    """
    A = check_matrix(A)
    independence_tol = ZERO_TOL if independence_tol is None else independence_tol
    Q, R = _qr_gram_schmidt(A, independence_tol=independence_tol)
    if drop_columns:
        return Q[:, np.diagonal(R) >= independence_tol]
    return Q


def column_space_projector(
    A: Matrix, independence_tol: Optional[float] = None
) -> NPMatrix:
    """Retorna uma matriz que projeta vetores para o espaço de colunas de `A`.

    Útil para soluções least squares de sistemas lineares.

    Entrada:
        `A (Matrix)` - Matriz cujas colunas definem o espaço de projeção.
        `independence_tol (None | float)` - Tolerância para considerar que uma coluna de `A` colapsou na origem (é combinação de outras colunas de `A`). Esse parâmetro é repassado para o método `rank_revealing_qr`.

    Saída:
        `NPMatrix` - Matriz que projeta vetores para o espaço de colunas definido pela matriz `A`.
    """
    independence_tol = ZERO_TOL if independence_tol is None else independence_tol
    _, Q, _ = rank_revealing_qr(A, independence_tol=independence_tol)
    if not Q.size:
        raise ValueError("All columns are collapsed onto the zero vector")
    return Q.dot(Q.T)
