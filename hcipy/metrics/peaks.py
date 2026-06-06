from .._math.backends import array_namespace
from ..field import Field
from ..fourier import FastFourierTransform

__all__ = [
    'sub_pixel_peak',
    'centroid',
    'fwhm',
    'ellipticity',
    'image_shift',
]

def sub_pixel_peak(image, mask=None):
    '''Locate the peak of a 2-D image with sub-pixel precision.

    A quadratic surface is fit to the 3x3 neighborhood of the integer
    peak position. If the integer peak is on the boundary of the image,
    or if the quadratic fit does not have a maximum (concave down), the
    integer peak position is returned.

    Parameters
    ----------
    image : Field
        A two-dimensional, real-valued image defined on a separated
        Cartesian grid.
    mask : Field or None
        An optional boolean array (same shape as `image.shaped`) that
        restricts the peak search. Pixels where `mask` is False are
        treated as ``-inf`` for the purpose of finding the maximum.

    Returns
    -------
    x_peak : scalar
        The x-coordinate of the peak in grid units.
    y_peak : scalar
        The y-coordinate of the peak in grid units.
    '''
    if image.dtype.kind == 'c':
        raise TypeError('sub_pixel_peak requires a real-valued image. '
                        'Pass xp.abs(image)**2 explicitly.')

    xp = array_namespace(image)
    shp = image.shaped
    Ny, Nx = shp.shape

    if mask is not None:
        shp_search = xp.where(mask, shp, -xp.inf)
    else:
        shp_search = shp

    peak = int(xp.argmax(shp_search))
    i, j = peak // Nx, peak % Nx

    x = image.grid.separated_coords[0]
    y = image.grid.separated_coords[1]
    x_peak = float(x[j])
    y_peak = float(y[i])

    if 0 < i < Ny - 1 and 0 < j < Nx - 1:
        patch = shp[i - 1:i + 2, j - 1:j + 2]
        a_row = (patch[0, 1] - 2.0 * patch[1, 1] + patch[2, 1]) / 2.0
        a_col = (patch[1, 0] - 2.0 * patch[1, 1] + patch[1, 2]) / 2.0

        if a_row < 0.0 and a_col < 0.0:
            delta_x = (x[1] - x[0]) if x.shape[0] > 1 else 1.0
            delta_y = (y[1] - y[0]) if y.shape[0] > 1 else 1.0
            j_offset = -(patch[1, 2] - patch[1, 0]) / (4.0 * a_col)
            i_offset = -(patch[2, 1] - patch[0, 1]) / (4.0 * a_row)
            return x_peak + float(j_offset * delta_x), y_peak + float(i_offset * delta_y)

    return x_peak, y_peak

def centroid(image, mask=None):
    '''Compute the intensity-weighted center of mass of a 2-D image.

    Parameters
    ----------
    image : Field
        A two-dimensional, real-valued image defined on a separated
        Cartesian grid.
    mask : Field or None
        An optional boolean array (same shape as `image.shaped`) that
        restricts the pixels that contribute to the centroid.

    Returns
    -------
    x_c : scalar
        The x-coordinate of the centroid in grid units.
    y_c : scalar
        The y-coordinate of the centroid in grid units.

    Raises
    ------
    ValueError
        If the total masked intensity is zero.
    '''
    xp = array_namespace(image)
    if image.dtype.kind == 'c':
        raise TypeError('centroid requires a real-valued image. '
                        'Pass xp.abs(image)**2 explicitly.')

    if mask is not None:
        image = Field(xp.where(mask, image.shaped, 0).ravel(), image.grid)

    total = xp.sum(image)

    x_c = xp.sum(image * image.grid.x) / total
    y_c = xp.sum(image * image.grid.y) / total

    return x_c, y_c

def fwhm(image, method='radial', cx=None, cy=None):
    '''Measure the full width at half maximum of a 2-D image.

    Currently only ``method='radial'`` is supported: the image is binned
    by radius from `(cx, cy)` and the FWHM is found by linear
    interpolation of the half-maximum crossing in the resulting radial
    profile.

    Parameters
    ----------
    image : Field
        A two-dimensional, real-valued image defined on a separated
        Cartesian grid.
    method : {'radial'}
        The method to use for FWHM extraction.
    cx, cy : scalar or None
        The center around which to compute the radial profile, in grid
        units. If either is None, the center defaults to
        :func:`sub_pixel_peak`.

    Returns
    -------
    scalar
        The FWHM in grid units. Returns ``inf`` if the radial profile
        never drops below half of its maximum value.
    '''
    if method != 'radial':
        raise ValueError("Only method='radial' is supported.")

    xp = array_namespace(image)
    if image.dtype.kind == 'c':
        raise TypeError('fwhm requires a real-valued image. '
                        'Pass xp.abs(image)**2 explicitly.')

    if cx is None or cy is None:
        cx, cy = sub_pixel_peak(image)

    x = image.grid.x
    y = image.grid.y
    r = xp.sqrt((x - cx) ** 2 + (y - cy) ** 2).ravel()
    img1d = image.ravel()

    sep_x = image.grid.separated_coords[0]
    sep_y = image.grid.separated_coords[1]
    delta_r = min(abs(sep_x[1] - sep_x[0]), abs(sep_y[1] - sep_y[0]))
    if delta_r <= 0:
        return xp.inf

    r_max = xp.max(r)
    n_bins = int(r_max / delta_r) + 1
    bin_edges = xp.linspace(0.0, r_max, n_bins + 1)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2.0

    bin_idx = xp.searchsorted(bin_edges, r, side='right') - 1
    bin_idx = xp.clip(bin_idx, 0, n_bins - 1)

    profile = []
    for b in range(n_bins):
        in_bin = bin_idx == b
        count = xp.sum(in_bin)
        if count > 0:
            value = xp.sum(xp.where(in_bin, img1d, 0)) / count
        else:
            value = 0.0
        profile.append(value)

    peak_value = max(profile)
    half = peak_value / 2.0

    k = 0
    found = False
    for idx, p in enumerate(profile):
        if p < half:
            k = idx
            found = True
            break
    if not found:
        return xp.inf
    if k == 0:
        return 0.0

    r0 = bin_centers[k - 1]
    r1 = bin_centers[k]
    p0 = profile[k - 1]
    p1 = profile[k]
    if p0 == p1:
        return r0
    r_half = r0 + (half - p0) * (r1 - r0) / (p1 - p0)
    return 2.0 * r_half

def ellipticity(image, cx=None, cy=None, mask=None):
    '''Compute the second-moment ellipticity of a 2-D image.

    The 2x2 second-moment matrix of the image is::

        M_xx = sum(I * dx**2) / sum(I)
        M_yy = sum(I * dy**2) / sum(I)
        M_xy = sum(I * dx * dy) / sum(I)

    where ``dx = x - cx``, ``dy = y - cy``. The eigenvalues of ``M``
    give the principal second moments, and the semi-axes of the
    equivalent ellipse are their square roots. The position angle is
    computed analytically from the matrix elements.

    Parameters
    ----------
    image : Field
        A two-dimensional, real-valued image defined on a separated
        Cartesian grid.
    cx, cy : scalar or None
        The center around which the moments are computed, in grid units.
        If either is None, the center defaults to :func:`sub_pixel_peak`.
    mask : Field or None
        An optional boolean array (same shape as `image.shaped`) that
        restricts the pixels that contribute to the moments.

    Returns
    -------
    a : scalar
        The semi-major axis length, in grid units.
    b : scalar
        The semi-minor axis length, in grid units.
    theta : scalar
        The position angle of the major axis, in radians, in
        ``[-pi/2, pi/2]``.
    ell : scalar
        The ellipticity ``1 - b / a``.
    '''
    xp = array_namespace(image)
    if image.dtype.kind == 'c':
        raise TypeError('ellipticity requires a real-valued image. '
                        'Pass xp.abs(image)**2 explicitly.')

    if cx is None or cy is None:
        cx, cy = sub_pixel_peak(image)

    shp = image.shaped
    if mask is not None:
        shp = xp.where(mask, shp, 0)

    total = xp.sum(shp)
    if total == 0:
        raise ValueError('The masked image has zero total intensity.')

    dx = (image.grid.x - cx).shaped
    dy = (image.grid.y - cy).shaped
    M_xx = xp.sum(shp * dx * dx) / total
    M_yy = xp.sum(shp * dy * dy) / total
    M_xy = xp.sum(shp * dx * dy) / total

    eigvals = xp.linalg.eigvalsh(xp.asarray([[M_xx, M_xy], [M_xy, M_yy]]))
    sorted_eigvals = xp.sort(eigvals)[::-1]
    a = xp.sqrt(sorted_eigvals[0])
    b = xp.sqrt(sorted_eigvals[1])
    theta = 0.5 * xp.atan2(2.0 * M_xy, M_xx - M_yy)
    ell = 0.0 if a == 0 else 1.0 - b / a
    return a, b, theta, ell

def image_shift(image, reference):
    '''Estimate the sub-pixel shift that aligns `reference` to `image`.

    The shift is computed by cross-correlating the two images in
    Fourier space and locating the correlation peak with sub-pixel
    accuracy via a 1D parabolic fit on the 3x3 cyclic neighborhood
    of the peak in the unshifted cross-correlation. The returned
    shift is in grid units and follows the convention that
    ``reference`` translated by ``(dx, dy)`` aligns with `image`.

    Parameters
    ----------
    image : Field
        A two-dimensional, real-valued image defined on a separated
        Cartesian grid.
    reference : Field
        A two-dimensional, real-valued image on the same grid as
        `image`.

    Returns
    -------
    dx : scalar
        The x-shift in grid units (positive moves `reference` to the
        right).
    dy : scalar
        The y-shift in grid units (positive moves `reference` up).
    '''
    if image.dtype.kind == 'c' or reference.dtype.kind == 'c':
        raise TypeError('image_shift requires real-valued images. '
                        'Pass xp.abs(image)**2 explicitly.')

    fft = FastFourierTransform(image.grid)
    F_A = fft.forward(image)
    F_B = fft.forward(reference)
    R = F_A * F_B.conj()
    C = fft.backward(R)

    x_peak, y_peak = sub_pixel_peak(abs(C))

    return x_peak, y_peak
