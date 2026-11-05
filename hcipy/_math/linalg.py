import numpy as np
from scipy.linalg import get_blas_funcs
from array_api_compat import is_numpy_namespace

from .backends import array_namespace


def matmul(a, b, *, alpha=1, adjoint_a=False, adjoint_b=False, out=None):
    '''Multiply matrices, optionally applying adjoints and a scalar factor.

    Leading dimensions are broadcast as batch dimensions. Both inputs must have
    at least two dimensions and the same floating-point or complex dtype.

    Parameters
    ----------
    a, b : Array
        The matrices to multiply.
    alpha : scalar, optional
        A scalar multiplier for the product.
    adjoint_a, adjoint_b : bool, optional
        Whether to conjugate-transpose the respective matrix.
    out : Array or None, optional
        Optional output storage. NumPy uses this buffer when its shape and dtype
        match. Other backends allocate the result through their native matmul.
        Always use the returned array, which may differ from this buffer.

    Returns
    -------
    Array
        The scaled matrix product.
    '''
    xp = array_namespace(a, b)
    if is_numpy_namespace(xp):
        shape = np.broadcast_shapes(a.shape[:-2], b.shape[:-2])
        output_shape = shape + (a.shape[-1 if adjoint_a else -2], b.shape[-2 if adjoint_b else -1])
        if out is None or out.shape != output_shape or out.dtype != a.dtype:
            out = np.empty(output_shape, dtype=a.dtype)

        a = np.broadcast_to(a, shape + a.shape[-2:])
        b = np.broadcast_to(b, shape + b.shape[-2:])
        gemm = get_blas_funcs('gemm', dtype=a.dtype)
        for i in np.ndindex(shape):
            # C-ordered matrices are Fortran ordered when transposed.
            result = gemm(alpha, b[i].T, a[i].T, trans_a=2 if adjoint_b else 0, trans_b=2 if adjoint_a else 0, c=out[i].T, overwrite_c=True)
            if not np.shares_memory(result, out[i]):
                out[i] = result.T
        return out

    if adjoint_a:
        a = xp.conj(xp.matrix_transpose(a))
    if adjoint_b:
        b = xp.conj(xp.matrix_transpose(b))
    return alpha * xp.matmul(a, b)
