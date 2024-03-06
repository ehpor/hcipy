import numpy as np
from ..field import Field

def large_poisson(lam, thresh=1e6):
    """
    Draw samples from a Poisson distribution, taking care of large values of `lam`.

    At large values of `lam` the distribution automatically switches to the corresponding normal distribution.
    This switch is independently decided for each expectation value in the `lam` array.

    Parameters
    ----------
    lam : array_like
        Expectation value for the Poisson distribution. Must be >= 0.
    thresh : float
        The threshold at which the distribution switched from a Poisson to a normal distribution.

    Returns
    -------
    array_like
        The drawn samples from the Poisson or normal distribution, depending on the expectation value.
    """
    large = lam > thresh
    small = ~large

    # Use normal approximation if the number of photons is large
    n = np.zeros(lam.shape)
    n[large] = np.round(lam[large] + np.random.normal(size=np.sum(large)) * np.sqrt(lam[large]))
    n[small] = np.random.poisson(lam[small], size=np.sum(small))

    if hasattr(lam, 'grid'):
        n = Field(n, lam.grid)

    return n

def large_gamma(lam, theta, thresh=1e6):
    """
    Draw samples from a Gamma distribution, taking care of large values of `lam`.

    At large values of `lam` the distribution automatically switches to the corresponding normal distribution.
    This switch is independently decided for each expectation value in the `lam` array.

    Parameters
    ----------
    lam : array_like
        The shape parameter distribution. Must be >= 0.
    theta : array_like
        The scale parameter of the Gamma distribution.
    thresh : float
        The threshold at which the distribution switched from a Gamma to a normal distribution.

    Returns
    -------
    array_like
        The drawn samples from the Gamma or normal distribution, depending on the expectation value.
    """
    large = lam > thresh
    small = ~large

    theta = theta * np.ones_like(lam)

    # Use normal approximation if the shape parameter is large
    n = np.zeros(lam.shape)
    mean = lam * theta
    std = np.sqrt(mean * theta)

    n[large] = mean[large] + std[large] * np.random.normal(size=np.sum(large))
    n[small] = np.random.gamma(lam[small], theta[small], size=np.sum(small))

    if hasattr(lam, 'grid'):
        n = Field(n, lam.grid)

    return n

def make_emccd_noise(photo_electron_flux, read_noise, emgain):
    """
    Draw samples from a Poisson-Gamma-Normal distribution, which is an accurate model of EMCCDs (following [Hirsch2013]_).

    The model is most accurate for high emgains.

    .. [Hirsch2013] Michael Hirsch et al. " A stochastic model for electron multiplication
        charge-coupled devices - from theory to practice." PloS one 8.1 (2013): e53671.

    Parameters
    ----------
    photo_electron_flux : array_like
        The incoming photo-electron flux, which is usually a combination of QE * (photon_flux + background) + dark current.
    read_noise : array_like
        The read noise of the camera.
    emgain : float
        The electron multiplying gain of the EMCCD process. Must be larger than 0.

    Returns
    -------
    array_like
        The noisy realization of the EMCCD detection.
    """
    photo_electrons = large_poisson(photo_electron_flux)

    detector_counts = large_gamma(photo_electrons, emgain) + read_noise * np.random.standard_normal(photo_electron_flux.shape)

    return detector_counts
