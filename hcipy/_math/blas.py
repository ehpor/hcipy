import functools

import numpy as np
from scipy.linalg import get_blas_funcs
from array_api_compat import is_numpy_namespace

from .backends import array_namespace


@functools.lru_cache(maxsize=None)
def _get_numpy_gemm(dtype):
    return get_blas_funcs('gemm', dtype=dtype)


def _apply_transpose(x, trans, xp):
    '''Apply a BLAS-style transpose to the last two axes of an array.'''
    if trans == 0:
        return x

    if trans not in (1, 2):
        raise ValueError('trans must be 0, 1 or 2.')

    if x.ndim < 2:
        raise ValueError('Cannot transpose an array with fewer than two dimensions.')

    axes = tuple(range(x.ndim - 2)) + (x.ndim - 1, x.ndim - 2)
    x = xp.permute_dims(x, axes)

    if trans == 2:
        x = xp.conj(x)

    return x


def gemm(alpha, a, b, *, beta=None, c=None, trans_a=0, trans_b=0, overwrite_c=False, xp=None):
    '''Compute a general matrix-matrix product.

    This function computes

    .. math:: C = \\alpha \\, op(A) \\, op(B) + \\beta \\, C,

    where ``op()`` is the identity, transpose or conjugate transpose, selected
    by `trans_a` and `trans_b`. If `alpha` is None, the product is not scaled.
    If `beta` is None, the `c` term is omitted.

    On NumPy arrays this dispatches to the BLAS ``gemm`` implementation from
    SciPy. On other Array API backends it is implemented using `xp.matmul`.

    Parameters
    ----------
    alpha : scalar or None
        The scalar multiplying the matrix product. If None, no scaling is
        applied.
    a : Array
        The first input matrix, with at least two dimensions.
    b : Array
        The second input matrix, with at least two dimensions.
    beta : scalar, optional
        The scalar multiplying `c`. If None, the `c` term is omitted.
    c : Array, optional
        The matrix to which the scaled product is added. If `overwrite_c` is
        True, it may also be used to store the result.
    trans_a : int, optional
        Whether to use `a`, its transpose (1), or its conjugate transpose (2).
    trans_b : int, optional
        Whether to use `b`, its transpose (1), or its conjugate transpose (2).
    overwrite_c : bool, optional
        Whether `c` may be overwritten with the result. This is only supported
        on the NumPy backend, where `c` must be C-contiguous.
    xp : module, optional
        The Array API namespace. If None, it is inferred from `a`, `b` and `c`.

    Returns
    -------
    Array
        The resulting matrix product.
    '''
    if trans_a not in (0, 1, 2):
        raise ValueError('trans_a must be 0, 1 or 2.')

    if trans_b not in (0, 1, 2):
        raise ValueError('trans_b must be 0, 1 or 2.')

    if xp is None:
        xp = array_namespace(a, b, c)

    if is_numpy_namespace(xp) and a.ndim == 2 and b.ndim == 2:
        if c is not None and beta is None and not overwrite_c:
            c = None

        dtype = c.dtype if c is not None else a.dtype
        gemm_func = _get_numpy_gemm(np.dtype(dtype))

        result = gemm_func(
            1.0 if alpha is None else alpha, b.T, a.T,
            beta=0.0 if beta is None else beta,
            c=None if c is None else c.T,
            trans_a=trans_b, trans_b=trans_a, overwrite_c=overwrite_c,
        )

        if c is not None and overwrite_c:
            return c

        return result.T

    if c is not None and overwrite_c and not is_numpy_namespace(xp):
        raise NotImplementedError('overwrite_c is only supported on the NumPy backend.')

    a = _apply_transpose(a, trans_a, xp)
    b = _apply_transpose(b, trans_b, xp)

    result = xp.matmul(a, b)

    if alpha is not None:
        result = alpha * result

    if c is not None and beta is not None:
        result = result + beta * c

    if c is not None and overwrite_c:
        c[...] = result
        return c

    return result
