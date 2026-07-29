import numpy as np
from numba import njit, prange
from array_api_compat import is_numpy_array, is_jax_array


def _quintic_weights(shift):
    """Compute the 7-element quintic B-spline kernel for a fractional shift.

    Parameters
    ----------
    shift : float
        Fractional shift in ``[-0.5, 0.5]``.  The sign determines the
        padding direction of the kernel.

    Returns
    -------
    ndarray of shape (7,)
        The 7-element 1-D convolution kernel.
    """
    f = abs(shift)
    b = np.empty(6)
    for i, offset in enumerate([2.0 + f, 1.0 + f, f, 1.0 - f, 2.0 - f, 3.0 - f]):
        s = 0.0
        x = offset + 3.0
        for c in (1.0, -6.0, 15.0, -20.0, 15.0, -6.0, 1.0):
            if x > 0.0:
                s += c * (x ** 5)
            x -= 1.0
        b[i] = s / 120.0

    if shift >= 0:
        return np.asarray([0.0, b[0], b[1], b[2], b[3], b[4], b[5]], dtype=b.dtype)
    else:
        return np.asarray([b[5], b[4], b[3], b[2], b[1], b[0], 0.0], dtype=b.dtype)


@njit(parallel=True, fastmath=True, cache=True)
def _row_pass(img, kernel, radius, out):
    """Numba-parallel horizontal (row) pass of a separable convolution.

    For each output pixel ``(y, x)``:

    .. math::

        \\mathtt{out}[y, x] = \\sum_{k=-\\mathtt{radius}}^{\\mathtt{radius}}
        \\mathtt{kernel}[k + \\mathtt{radius}] \\cdot
        \\mathtt{img}[y, \\operatorname{clamp}(x + k, 0, W-1)]

    Out-of-bounds indices are clamped to the nearest edge (``mode='nearest'``).

    Parameters
    ----------
    img : ndarray of shape (H, W)
        Input image.
    kernel : ndarray of shape (2 * radius + 1,)
        1-D convolution kernel.
    radius : int
        Half-width of the kernel (``len(kernel) // 2``).
    out : ndarray of shape (H, W)
        Output buffer, overwritten in place.
    """
    H, W = img.shape
    Wm1 = W - 1

    for y in prange(H):
        row = img[y]
        for x in range(W):
            acc = 0.0

            for k in range(-radius, radius + 1):
                xx = max(0, min(Wm1, x + k))

                acc += row[xx] * kernel[k + radius]

            out[y, x] = acc


@njit(parallel=True, fastmath=True, cache=True)
def _col_pass(img, kernel, radius, out):
    """Numba-parallel vertical (column) pass of a separable convolution.

    For each output pixel ``(y, x)``:

    .. math::

        \\mathtt{out}[y, x] = \\sum_{k=-\\mathtt{radius}}^{\\mathtt{radius}}
        \\mathtt{kernel}[k + \\mathtt{radius}] \\cdot
        \\mathtt{tmp}[\\operatorname{clamp}(y + k, 0, H-1), x]

    Out-of-bounds indices are clamped to the nearest edge (``mode='nearest'``).

    Parameters
    ----------
    img : ndarray of shape (H, W)
        Intermediate image from the horizontal pass.
    kernel : ndarray of shape (2 * radius + 1,)
        1-D convolution kernel.
    radius : int
        Half-width of the kernel (``len(kernel) // 2``).
    out : ndarray of shape (H, W)
        Output buffer, overwritten in place.
    """
    H, W = img.shape
    Hm1 = H - 1

    for y in prange(H):
        for x in range(W):
            acc = 0.0

            for k in range(-radius, radius + 1):
                yy = max(0, min(Hm1, y + k))

                acc += img[yy, x] * kernel[k + radius]

            out[y, x] = acc


def separable_convolve_numpy(img, kernel_row, kernel_col):
    """Numba-accelerated separable 2-D convolution.

    Only accepts :class:`numpy.ndarray` inputs.  For other array backends
    (CuPy, JAX, PyTorch, …) use :func:`separable_convolve_fallback`.

    Parameters
    ----------
    img : ndarray of shape (H, W)
        Input image.
    kernel_row : array-like of shape (K,)
        1-D kernel applied along columns (axis 1).  Must have odd length.
    kernel_col : array-like of shape (K,)
        1-D kernel applied along rows (axis 0).  Must have odd length.

    Returns
    -------
    out : ndarray of shape (H, W)
        Convolved image.

    Raises
    ------
    ValueError
        If either kernel has even length.
    """
    kernel_row = np.asarray(kernel_row, dtype=img.dtype)
    kernel_col = np.asarray(kernel_col, dtype=img.dtype)

    radius_x = len(kernel_row) // 2
    radius_y = len(kernel_col) // 2

    if len(kernel_row) != 2 * radius_x + 1:
        raise ValueError("kernel_row must have odd length")

    if len(kernel_col) != 2 * radius_y + 1:
        raise ValueError("kernel_col must have odd length")

    img = np.ascontiguousarray(img)
    tmp = np.empty_like(img)
    out = np.empty_like(img)

    _row_pass(img, kernel_row, radius_x, tmp)
    _col_pass(tmp, kernel_col, radius_y, out)

    return out


def _iadd(a, indices, b):
    """Accumulate ``b`` into ``a[indices]``, returning the result.

    Uses ``a.at[indices].add(b)`` for array backends that expose an ``.at``
    property (e.g. JAX), and in-place ``a[indices] += b`` for all others
    (NumPy, CuPy, PyTorch, …).

    Parameters
    ----------
    a : array
        Array to accumulate into.
    indices : tuple of slice or int
        Indexing tuple understood by the backend.
    b : array
        Values to add.

    Returns
    -------
    array
        ``a`` after the accumulation (possibly a new array for immutable
        backends).
    """
    if is_jax_array(a):
        return a.at[indices].add(b)

    a[indices] += b
    return a


def separable_convolve_fallback(img, kernel_row, kernel_col):
    """Separable 2-D convolution via slice-based weighted accumulation.

    Expressed as view-based weighted sums (no ``take(…, mode='clip')``) for
    maximum Array-API portability.  Works on any array backend (NumPy, CuPy,
    JAX, PyTorch, …) without modification.

    The convolution is split into two 1-D passes:

    1. **Horizontal pass**: each row is convolved with *kernel_row*.
    2. **Vertical pass**: each column of the result is convolved with
       *kernel_col*.

    Boundary conditions are ``mode='nearest'`` (clamp to edge).

    Parameters
    ----------
    img : array of shape (H, W)
        Input image.
    kernel_row : array-like of shape (K,)
        1-D kernel applied along columns (axis 1).  Must have odd length.
    kernel_col : array-like of shape (K,)
        1-D kernel applied along rows (axis 0).  Must have odd length.

    Returns
    -------
    out : array of shape (H, W)
        Convolved image, produced by the same backend as *img*.
    """
    H, W = img.shape[0], img.shape[1]

    rx = len(kernel_row) // 2
    ry = len(kernel_col) // 2

    # ---- Horizontal pass ----
    tmp = img * float(kernel_row[rx])
    for i in range(kernel_row.shape[0]):
        w = kernel_row[i]
        s = i - rx

        if s > 0:
            tmp = _iadd(tmp, (slice(None), slice(None, W - s)),
                        img[:, s:] * w)
            tmp = _iadd(tmp, (slice(None), slice(W - s, None)),
                        img[:, -1:] * w)
        elif s < 0:
            tmp = _iadd(tmp, (slice(None), slice(-s, None)),
                        img[:, :W + s] * w)
            tmp = _iadd(tmp, (slice(None), slice(None, -s)),
                        img[:, :1] * w)

    # ---- Vertical pass ----
    out = tmp * float(kernel_col[ry])
    for i in range(kernel_col.shape[0]):
        w = kernel_col[i]
        s = i - ry

        if s > 0:
            out = _iadd(out, (slice(None, H - s), slice(None)),
                        tmp[s:, :] * w)
            out = _iadd(out, (slice(H - s, None), slice(None)),
                        tmp[-1:, :] * w)
        elif s < 0:
            out = _iadd(out, (slice(-s, None), slice(None)),
                        tmp[:H + s, :] * w)
            out = _iadd(out, (slice(None, -s), slice(None)),
                        tmp[:1, :] * w)

    return out


def separable_convolve(img, kernel_row, kernel_col):
    """Separable 2-D convolution with automatic backend dispatch.

    Dispatches to :func:`separable_convolve_numpy` for ``numpy.ndarray``
    inputs, and to :func:`separable_convolve_fallback` for all other array
    types (CuPy, JAX, PyTorch, …).

    Parameters
    ----------
    img : array of shape (H, W)
        Input image.
    kernel_row : array-like of shape (K,)
        1-D kernel applied along columns (axis 1).  Must have odd length.
    kernel_col : array-like of shape (K,)
        1-D kernel applied along rows (axis 0).  Must have odd length.

    Returns
    -------
    out : array of shape (H, W)
        Convolved image, produced by the same backend as *img*.
    """
    if is_numpy_array(img):
        return separable_convolve_numpy(img, kernel_row, kernel_col)
    else:
        return separable_convolve_fallback(img, kernel_row, kernel_col)


def subpixel_shift(img, row_shift, col_shift):
    """Sub-pixel shift via separable 5th-order B-spline convolution.

    The image is shifted by the specified fractional amounts using quintic
    B-spline interpolation (no IIR pre-filter — acceptable for band-limited
    atmospheric data).

    ``row_shift`` and ``col_shift`` **must** be in ``[-0.5, 0.5]``.
    The caller is responsible for removing integer-pixel shifts (e.g. via
    :func:`numpy.roll` or :func:`numpy.roll` equivalents on other backends)
    before calling this function.

    Parameters
    ----------
    img : array of shape (H, W)
        Input image.
    row_shift : float
        Fractional shift along axis 0 (rows).  Must be in ``[-0.5, 0.5]``.
    col_shift : float
        Fractional shift along axis 1 (columns).  Must be in ``[-0.5, 0.5]``.

    Returns
    -------
    out : array of shape (H, W)
        Sub-pixel shifted image, produced by the same backend as *img*.
    """
    wy = _quintic_weights(row_shift)
    wx = _quintic_weights(col_shift)

    return separable_convolve(img, wx, wy)
