import numpy as np
from math import pi

def binned_profile(y, x, bins=20, statistic='mean'):
    '''Create a profile of y(x) in the specified bins.

    Parameters
    ----------
    y : array_like
        The y-coordinates of the data points. This should be 1-dimensional.
    x : array_like
        The x-coordinates of the data points. This should be 1-dimensional
        and the same size as `y`.
    bins : array_like or integer
        The bin edges of the profile. If this is an integer, `bins` is the
        number of bins that will be equally distributed along the whole range
        of `x`.
    statistic : string
        The statistic to compute (default is 'mean').
        The following statistics are available:
        * 'mean': compute the mean of values for points within the bin edges while ignoring NaN values.
        * 'sum': compute the sum of values for points within the bin edges while ignoring NaN values.
        * 'min': compute the minimum of values for points within the bin edges while ignoring NaN values.
        * 'max': compute the maximum of values for points within the bin edges while ignoring NaN values.
        * 'std': compute the standard deviation of values for points within the bin edges while ignoring NaN values.
        * 'var': compute the variance of values for points within the bin edges while ignoring NaN values.

    Returns
    -------
    bin_centers : array_like
        The center of each of the bins.
    profile : array_like
        The y-values of the resulting profile.
    std_profile : array_like
        The standard deviation within each bin.
    num_per_bin : array_like
        The number of samples per bin.

    Raises
    ------
    RuntimeError
        If the number of bins are negative or zero.
    ValueError
        If the statistic is not implemented.
    '''
    if np.isscalar(bins):
        if bins <= 0:
            raise RuntimeError('The number of bins should be positive.')

        # Equally space the bins.
        bins = np.linspace(np.min(x), np.max(x), bins + 1)

    bin_centers = (bins[1:] + bins[:-1]) / 2
    num_bins = len(bin_centers)

    num_per_bin = np.histogram(x, bins)[0]
    which_bin = np.digitize(x, bins)

    statistics = {'mean': np.nanmean, 'median': np.nanmedian, 'max': np.nanmax, 'min': np.nanmin, 'std': np.nanstd, 'var': np.nanvar}

    if statistic not in statistics:
        raise ValueError('Statistic %s not implemented.' % statistic)

    func = statistics[statistic]

    profile = np.array([func(y[which_bin == b]) for b in range(1, num_bins + 1)])
    std_profile = np.array([np.nanstd(y[which_bin == b]) for b in range(1, num_bins + 1)])

    return bin_centers, profile, std_profile, num_per_bin

def azimutal_profile(image, num_bins, statistic='mean'):
    '''Create an azimuthal profile of the image around its center.

    Parameters
    ----------
    image : Field
        The image that we want an azimuthal profile from. This image must be
        two-dimensional.
    num_bins : integer
        The number of bins in theta. Bins will be equally distributed in theta.
    statistic : string
        The statistic to compute (default is 'mean').
        The following statistics are available:
        * 'mean': compute the mean of values for points within the bin edges.
        * 'sum': compute the sum of values for points within the bin edges.
        * 'min': compute the minimum of values for points within the bin edges.
        * 'max': compute the maximum of values for point within the bin edges.

    Returns
    -------
    bin_centers : array_like
        The center of each of the bins.
    profile : array_like
        The y-values of the resulting azimuthal profile.
    std_profile : array_like
        The standard deviation within each bin.
    num_per_bin : array_like
        The number of samples per bin.

    Raises
    ------
    RuntimeError
        If the number of bins are negative or zero.
    ValueError
        If the statistic is not implemented.
    '''
    theta = image.grid.as_('polar').theta
    bins = np.linspace(-pi, pi, num_bins + 1)

    return binned_profile(image, theta, bins, statistic=statistic)

def radial_profile(image, bin_size, statistic='mean'):
    '''Create a radial profile of the image around its center.

    Parameters
    ----------
    image : Field
        The image that we want a radial profile from. This image must be
        two-dimensional.
    bin_size : scalar
        The extent of each bin. Each bin will be a ring from r to r+`bin_size`.
    statistic : string
        The statistic to compute (default is 'mean').
        The following statistics are available:
        * 'mean': compute the mean of values for points within the bin edges.
        * 'sum': compute the sum of values for points within the bin edges.
        * 'min': compute the minimum of values for points within the bin edges.
        * 'max': compute the maximum of values for point within the bin edges.

    Returns
    -------
    bin_centers : array_like
        The center of each of the bins.
    profile : array_like
        The y-values of the resulting radial profile.
    std_profile : array_like
        The standard deviation within each bin.
    num_per_bin : array_like
        The number of samples per bin.

    Raises
    ------
    ValueError
        If the statistic is not implemented.
    '''
    r = image.grid.as_('polar').r

    n_bins = int(np.ceil(r.max() / bin_size))
    max_bin = n_bins * bin_size
    bins = np.linspace(0, max_bin, n_bins + 1)

    return binned_profile(image, r, bins, statistic=statistic)
