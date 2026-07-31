import numba
import numexpr
import cmath
import numpy as np

from .backends import array_namespace
from array_api_compat import is_numpy_array, is_numpy_namespace

@numba.njit(parallel=True, fastmath=True, cache=True)
def _dft_matrix_regular_numpy(x0, u0, dx, du, Nx, Nu, dtype, out=None):
    """Compute a DFT matrix for equally spaced grids, using numba.

    The DFT matrix is defined as ``out[i, k] = exp(1j * u[k] * x[i])`` with
    ``x[i] = x0 + i * dx`` and ``u[k] = u0 + k * du``. Because both grids are
    equally spaced, the matrix can be factorised as

    .. math:: \\exp\\left(1j \\, u_k x_i\\right) = C \\, A_i \\, B_k \\, T_{i+k},

    where ``C`` is a constant and ``A``, ``B`` and ``T`` are precomputed
    arrays of length ``Nx``, ``Nu`` and ``Nx + Nu - 1`` respectively. This
    reduces the number of expensive complex exponential evaluations from
    ``Nx * Nu`` to ``Nx + Nu``; all other entries are obtained with complex
    multiplications. The rows of the output are computed in parallel.

    Parameters
    ----------
    x0 : float
        The starting coordinate of the ``x`` grid.
    u0 : float
        The starting coordinate of the ``u`` grid.
    dx : float
        The spacing of the ``x`` grid.
    du : float
        The spacing of the ``u`` grid.
    Nx : int
        The number of rows of the output matrix (the size of the ``x`` grid).
    Nu : int
        The number of columns of the output matrix (the size of the ``u`` grid).
    dtype : dtype
        The complex dtype of the output matrix.
    out : Array, optional
        An array of shape ``(Nx, Nu)`` and of the given `dtype` in which the
        result is stored. If not given, a new array is allocated.

    Returns
    -------
    Array
        The DFT matrix of shape ``(Nx, Nu)``.
    """
    alpha = du * dx / 2.0
    C = cmath.exp(1j * u0 * x0)

    A = np.empty(Nx, dtype=dtype)
    for i in range(Nx):
        A[i] = cmath.exp(1j * (u0 * dx * i - alpha * i * i))

    B = np.empty(Nu, dtype=dtype)
    for k in range(Nu):
        B[k] = cmath.exp(1j * (du * x0 * k - alpha * k * k))

    T = np.empty(Nx + Nu - 1, dtype=dtype)
    for n in range(Nx + Nu - 1):
        T[n] = cmath.exp(1j * (alpha * n * n))

    if out is None:
        out = np.empty((Nx, Nu), dtype=dtype)

    for i in numba.prange(Nx):
        ai = C * A[i]
        Ti = T[i:i + Nu]

        for k in range(Nu):
            out[i, k] = ai * B[k] * Ti[k]

    return out

def _dft_matrix_regular_fallback(x0, u0, dx, du, Nx, Nu, xp, dtype):
    """Compute a DFT matrix for equally spaced grids, using any Array API namespace.

    This is a fallback implementation of :func:`dft_matrix_regular` for
    non-NumPy backends. The grids ``x`` and ``u`` are reconstructed from their
    starting coordinates and spacings using the real dtype matching `dtype`,
    after which the matrix is evaluated directly as
    ``exp(1j * x[i] * u[k])``.

    Parameters
    ----------
    x0 : float
        The starting coordinate of the ``x`` grid.
    u0 : float
        The starting coordinate of the ``u`` grid.
    dx : float
        The spacing of the ``x`` grid.
    du : float
        The spacing of the ``u`` grid.
    Nx : int
        The number of rows of the output matrix (the size of the ``x`` grid).
    Nu : int
        The number of columns of the output matrix (the size of the ``u`` grid).
    xp : module
        The Array API namespace to use for the computation.
    dtype : dtype
        The complex dtype of the output matrix.

    Returns
    -------
    Array
        The DFT matrix of shape ``(Nx, Nu)`` and dtype ``dtype``.
    """
    float_dtype = xp.float32 if dtype == xp.complex64 else xp.float64

    x = xp.arange(Nx, dtype=float_dtype) * dx + x0
    u = xp.arange(Nu, dtype=float_dtype) * du + u0

    return xp.exp(1j * x[:, xp.newaxis] * u[xp.newaxis, :])

def dft_matrix_regular(x0, u0, dx, du, Nx, Nu, xp, dtype, conjugate=False, transpose=False, out=None):
    """Compute a DFT matrix for equally spaced grids.

    The DFT matrix is defined as ``out[i, k] = exp(1j * u[k] * x[i])`` with
    ``x[i] = x0 + i * dx`` and ``u[k] = u0 + k * du``. On the NumPy backend
    the matrix is computed with the numba implementation
    :func:`_dft_matrix_regular_numpy`; on other backends it is computed
    directly with :func:`_dft_matrix_regular_fallback`.

    Parameters
    ----------
    x0 : float
        The starting coordinate of the ``x`` grid.
    u0 : float
        The starting coordinate of the ``u`` grid.
    dx : float
        The spacing of the ``x`` grid.
    du : float
        The spacing of the ``u`` grid.
    Nx : int
        The number of rows of the output matrix (the size of the ``x`` grid).
    Nu : int
        The number of columns of the output matrix (the size of the ``u`` grid).
    xp : module
        The Array API namespace of the backend on which to compute the matrix.
    dtype : dtype
        The complex dtype of the output matrix.
    conjugate : bool
        If True, return the complex conjugate of the matrix.
    transpose : bool
        If True, return the transpose of the matrix. The output matrix then
        has shape ``(Nu, Nx)``.
    out : Array, optional
        An array of the right shape in which the result is stored. Only
        used on the NumPy backend; if not given, a new array is allocated.

    Returns
    -------
    Array
        The DFT matrix of shape ``(Nx, Nu)`` (or ``(Nu, Nx)`` if `transpose`
        is True).
    """
    if conjugate:
        x0 = -x0
        dx = -dx

    if transpose:
        x0, u0 = u0, x0
        dx, du = du, dx
        Nx, Nu = Nu, Nx

    if is_numpy_namespace(xp):
        return _dft_matrix_regular_numpy(x0, u0, dx, du, Nx, Nu, dtype, out=out)
    else:
        return _dft_matrix_regular_fallback(x0, u0, dx, du, Nx, Nu, xp, dtype)

def _dft_matrix_separated_numpy(x, u, out=None):
    """Compute a DFT matrix for separated grids, using numexpr.

    The DFT matrix is defined as ``out[i, k] = exp(1j * u[k] * x[i])`` for
    separated (not necessarily equally spaced) grids `x` and `u`. The
    computation is vectorized with numexpr.

    Parameters
    ----------
    x : Array
        The first grid, of shape ``(Nx,)``.
    u : Array
        The second grid, of shape ``(Nu,)``.
    out : Array, optional
        An array of shape ``(Nx, Nu)`` in which the result is stored. If not
        given, a new array is allocated.

    Returns
    -------
    Array
        The DFT matrix of shape ``(Nx, Nu)``.
    """
    complex_dtype = np.complex64 if x.dtype == np.float32 else np.complex128

    res = numexpr.evaluate('exp(1j * x * u)', {'x': x[:, np.newaxis], 'u': u[np.newaxis, :]}, out=out)
    return res.astype(complex_dtype, copy=False)

def _dft_matrix_separated_fallback(x, u):
    """Compute a DFT matrix for arbitrary grids, using any Array API namespace.

    This is a fallback implementation of :func:`dft_matrix_separated` for
    non-NumPy backends. The matrix is evaluated directly as
    ``exp(1j * x[i] * u[k])``.

    Parameters
    ----------
    x : Array
        The first grid, of shape ``(Nx,)``.
    u : Array
        The second grid, of shape ``(Nu,)``.

    Returns
    -------
    Array
        The DFT matrix of shape ``(Nx, Nu)``.
    """
    xp = array_namespace(x, u)

    return xp.exp(1j * x[:, xp.newaxis] * u[xp.newaxis, :])

def dft_matrix_separated(x, u, conjugate=False, transpose=False, out=None):
    """Compute a DFT matrix for arbitrary grids.

    The DFT matrix is defined as ``out[i, k] = exp(1j * u[k] * x[i])`` for
    arbitrary (not necessarily equally spaced) grids `x` and `u`. On the
    NumPy backend the matrix is computed with the numexpr implementation
    :func:`_dft_matrix_separated_numpy`; on other backends it is computed
    directly with :func:`_dft_matrix_separated_fallback`.

    Parameters
    ----------
    x : Array
        The first grid, of shape ``(Nx,)``.
    u : Array
        The second grid, of shape ``(Nu,)``.
    conjugate : bool
        If True, return the complex conjugate of the matrix.
    transpose : bool
        If True, return the transpose of the matrix. The output matrix then
        has shape ``(Nu, Nx)``.
    out : Array, optional
        An array of the right shape in which the result is stored. Only
        used on the NumPy backend; if not given, a new array is allocated.

    Returns
    -------
    Array
        The DFT matrix of shape ``(Nx, Nu)`` (or ``(Nu, Nx)`` if `transpose`
        is True).
    """
    if conjugate:
        x = -x

    if transpose:
        x, u = u, x

    if is_numpy_array(x):
        return _dft_matrix_separated_numpy(x, u, out=out)
    else:
        return _dft_matrix_separated_fallback(x, u)
