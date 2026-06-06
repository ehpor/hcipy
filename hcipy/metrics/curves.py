from .._math.backends import array_namespace
from .peaks import sub_pixel_peak

__all__ = [
    'encircled_energy',
    'ensquared_energy',
]


def encircled_energy(image, cx=None, cy=None, mask=None):
    '''Compute the encircled-energy curve of a 2-D image.

    Pixels are sorted by their radial distance from ``(cx, cy)`` and
    the cumulative intensity is normalized to one. The result is a
    step-like curve that is exact to within the image sampling.

    Parameters
    ----------
    image : Field
        A two-dimensional, real-valued image defined on a separated
        Cartesian grid.
    cx, cy : scalar or None
        The center in grid units. If either is None, the center
        defaults to :func:`sub_pixel_peak`.
    mask : Field or None
        An optional boolean array (same shape as `image.shaped`) that
        excludes pixels from the curve. Masked pixels are pushed to
        the end of the sort and contribute zero to the cumulative sum.

    Returns
    -------
    radii : ndarray
        1-D array of radial distances (in grid units) at which the
        encircled energy is evaluated, sorted in increasing order.
    ee : ndarray
        1-D array of encircled-energy values, normalized to one.
    '''
    xp = array_namespace(image)
    if image.dtype.kind == 'c':
        raise TypeError('encircled_energy requires a real-valued image. '
                        'Pass xp.abs(image)**2 explicitly.')

    if cx is None or cy is None:
        cx, cy = sub_pixel_peak(image)

    r = xp.sqrt((image.grid.x - cx) ** 2 + (image.grid.y - cy) ** 2)
    img1d = image.ravel()
    r1d = r.ravel()

    if mask is not None:
        m1d = mask.ravel()
        r1d = xp.where(m1d, r1d, xp.asarray(xp.inf))
        img1d = xp.where(m1d, img1d, 0)

    order = xp.argsort(r1d)
    sorted_r = r1d[order]
    sorted_I = img1d[order]

    cum = xp.cumulative_sum(sorted_I, include_initial=False)
    total = cum[-1]
    if total == 0:
        zeros = xp.zeros(sorted_r.shape, dtype=xp.asarray(0.0).dtype)
        return sorted_r, zeros

    return sorted_r, cum / total


def ensquared_energy(image, cx=None, cy=None, halfwidths=None):
    '''Compute the ensquared-energy curve of a 2-D image.

    For each halfwidth ``h`` in `halfwidths`, the fraction of total
    image intensity inside a square box of half-side ``h`` centered at
    ``(cx, cy)`` is computed.

    Parameters
    ----------
    image : Field
        A two-dimensional, real-valued image defined on a separated
        Cartesian grid.
    cx, cy : scalar or None
        The center in grid units. If either is None, the center
        defaults to :func:`sub_pixel_peak`.
    halfwidths : array_like or None
        A 1-D array of halfwidths (in grid units) at which to evaluate
        the ensquared energy. If None, defaults to 64 linearly-spaced
        values from 0 up to the maximum absolute x or y coordinate.

    Returns
    -------
    halfwidths : ndarray
        1-D array of halfwidths (in grid units).
    ee : ndarray
        1-D array of ensquared-energy values, normalized to one.
    '''
    xp = array_namespace(image)
    if image.dtype.kind == 'c':
        raise TypeError('ensquared_energy requires a real-valued image. '
                        'Pass xp.abs(image)**2 explicitly.')

    if cx is None or cy is None:
        cx, cy = sub_pixel_peak(image)

    img1d = image.ravel()
    total = xp.sum(img1d)

    if halfwidths is None:
        if total == 0:
            halfwidths = xp.linspace(0.0, 1.0, 64)
            return halfwidths, xp.zeros(halfwidths.shape)
        dx_max = xp.max(xp.abs(image.grid.x - cx))
        dy_max = xp.max(xp.abs(image.grid.y - cy))
        half_extent = max(dx_max, dy_max)
        halfwidths = xp.linspace(0.0, half_extent, 64)
    else:
        halfwidths = xp.asarray(halfwidths)

    if total == 0:
        return halfwidths, xp.zeros(halfwidths.shape)

    dx = xp.abs(image.grid.x - cx).ravel()
    dy = xp.abs(image.grid.y - cy).ravel()

    in_box = (dx[None, :] <= halfwidths[:, None]) & (dy[None, :] <= halfwidths[:, None])
    ee = xp.sum(in_box * img1d[None, :], axis=1) / total
    return halfwidths, ee
