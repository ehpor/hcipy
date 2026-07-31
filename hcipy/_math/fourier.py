import numba
import numexpr
import cmath
import numpy as np

from .backends import array_namespace
from array_api_compat import is_numpy_array, is_numpy_namespace

@numba.njit(parallel=True, fastmath=True, cache=True)
def dft_matrix_separated_numba(x0, u0, dx, du, Nx, Nu, dtype, out=None):
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
        Ti = T[i : i + Nu]

        for k in range(Nu):
            out[i, k] = ai * B[k] * Ti[k]

    return out

def dft_matrix_separated_fallback(x0, u0, dx, du, Nx, Nu, xp, dtype):
    x = xp.arange(Nx, dtype=dtype) * dx + x0
    u = xp.arange(Nu, dtype=dtype) * du + u0

    return xp.exp(1j * x[:, xp.newaxis] * u[xp.newaxis, :])

def dft_matrix_separated(x0, u0, dx, du, Nx, Nu, xp, dtype, conjugate=False, transpose=False, out=None):
    if conjugate:
        x0 = -x0
        dx = -dx

    if transpose:
        x0, u0 = u0, x0
        dx, du = du, dx
        Nx, Nu = Nu, Nx

    if is_numpy_namespace(xp):
        return dft_matrix_separated_numba(x0, u0, dx, du, Nx, Nu, dtype, out=out)
    else:
        return dft_matrix_separated_fallback(x0, u0, dx, du, Nx, Nu, xp, dtype)

def dft_matrix_unseparated_numba(x, u, out=None):
    return numexpr.evaluate('exp(1j * x * u)', {'x': x[:, np.newaxis], 'u': u[np.newaxis, :]}, out=out)

def dft_matrix_unseparated_fallback(x, u):
    xp = array_namespace(x, u)

    return xp.exp(1j * x[:, xp.newaxis] * u[xp.newaxis, :])

def dft_matrix_unseparated(x, u, conjugate=False, transpose=False, out=None):
    if conjugate:
        x = -x

    if transpose:
        x, u = u, x

    if is_numpy_array(x):
        return dft_matrix_unseparated_numba(x, u, out=out)
    else:
        return dft_matrix_unseparated_fallback(x, u)
