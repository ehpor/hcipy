from .atmospheric_model import Cn_squared_from_fried_parameter, MultiLayerAtmosphere
from .infinite_atmospheric_layer import InfiniteAtmosphericLayer

import numpy as np

def make_standard_atmospheric_layers(input_grid, L0=10):
    '''Create a standardized multi-layer atmosphere.

    The layer parameters are taken from [Guyon2005]_, in turn derived
    from [Tokovinin2005]_ and are representative for Mauna Kea, Hawaii.

    .. [Guyon 2005] Olivier Guyon, "Limits of Adaptive Optics for
        High-Contrast Imaging", ApJ 629 592 (2005).

    .. [Tokovinin2005] A. Tokovinin et al., "Optical Turbulence Profiles
        at Mauna Kea Measured by MASS and SCIDAR", PASP 117 395 (2005).

    Parameters
    ----------
    input_grid : Grid
        The input grid for the atmospheric layers.
    L0 : scalar
        The outer scale of the atmosphere

    Returns
    -------
    list
        A list of turbulence layers.
    '''
    heights = np.array([500, 1000, 2000, 4000, 8000, 16000])
    velocities = np.array([10, 10, 10, 10, 10, 10])
    Cn_squared = np.array([0.2283, 0.0883, 0.0666, 0.1458, 0.3350, 0.1350]) * 1e-12

    layers = []
    for h, v, cn in zip(heights, velocities, Cn_squared):
        layers.append(InfiniteAtmosphericLayer(input_grid, cn, L0, v, h, 2))

    return layers

def make_mauna_kea_atmospheric_layers(input_grid, cn_squared=None, outer_scale=None):
    return make_standard_atmospheric_layers(input_grid, outer_scale)

def make_las_campanas_atmospheric_layers(input_grid, r0=0.16, L0=25, wavelength=550e-9):
    '''Creates a multi-layer atmosphere for the Las Campanas Observatory site.

    The layer parameters are taken from [Males2019]_ who based it on site testing from [Prieto2010]_ and [Osip2011]_ .

    .. [Prieto2010] G. Prieto et al., “Giant Magellan telescope site testing seeing and
        turbulence statistics,” Proc. SPIE 7733, 77334O (2010).

    .. [Osip2011] Joanna E. Thomas-Osip et al. "Giant Magellan Telescope Site Testing Summary."
        arXiv:1101.2340 (2011).

    .. [Males2019] Jared Males et al. "Ground-based adaptive optics coronagraphic performance
        under closed-loop predictive control", JATIS, Volume 4, id. 019001 (2018).

    Parameters
    ----------
    input_grid : Grid
        The input grid for the atmospheric layers.
    r0 : scalar
        The integrated Cn^2 value for the atmosphere.
    L0 : scalar
        The outer scale of the atmosphere
    wavelength : scalar
        The wavelength in meters at which to calculate the Fried parameter (default: 550nm).

    Returns
    -------
    list
        A list of turbulence layers.
    '''
    heights = np.array([250, 500, 1000, 2000, 4000, 8000, 16000])
    velocities = np.array([10, 10, 20, 20, 25, 30, 25])

    integrated_cn_squared = Cn_squared_from_fried_parameter(r0, wavelength=500e-9)
    Cn_squared = np.array([0.42, 0.03, 0.06, 0.16, 0.11, 0.10, 0.12]) * integrated_cn_squared

    layers = []
    for h, v, cn in zip(heights, velocities, Cn_squared):
        layers.append(InfiniteAtmosphericLayer(input_grid, cn, L0, v, h, 2))

    return layers

def make_keck_atmospheric_layers(input_grid):
    '''Creates a multi-layer atmosphere for Keck Observatory.

    The atmospheric parameters are based off of [Keck AO note 303]_.

    .. [Keck AO note 303] https://www2.keck.hawaii.edu/optics/kpao/files/KAON/KAON303.pdf

    Parameters
    ----------
    input_grid : Grid
        The input grid of the atmospheric layers.

    Returns
    -------
    list
        A list of turbulence layers.
    '''
    heights = np.array([0, 2100, 4100, 6500, 9000, 12000, 14800])
    velocities = np.array([6.7, 13.9, 20.8, 29.0, 29.0, 29.0, 29.0])
    outer_scales = np.array([20, 20, 20, 20, 20, 20, 20])
    Cn_squared = np.array([0.369, 0.219, 0.127, 0.101, 0.046, 0.111, 0.027]) * 1e-12

    layers = []
    for h, v, cn, L0 in zip(heights, velocities, Cn_squared, outer_scales):
        layers.append(InfiniteAtmosphericLayer(input_grid, cn, L0, v, h, 2))

    return layers

def make_standard_atmosphere(input_grid, cn_squared=None, outer_scale=None, site='mauna_kea', **kwargs):
    '''Make a standard atmosphere for one of the built-in sites.

    Parameters
    ----------
    cn_squared : scalar or None
        The integrated Cn^2 value of the atmosphere. If this is None, the default,
        then the default Cn^2 of your requested site will be used.
    outer_scale : scalar or None
        The outer scale of all layers. If this is None, the default, then the default
        outer scale of your requested site will be used.
    site : {'mauna_kea', 'las_campanas', 'keck'}
        The site of the standard atmosphere. This has to be one of the implemented sites.
    **kwargs : kwargs
        Any additional kwargs will be fed through to the MultiLayerAtmosphere initializer.

    Returns
    -------
    MultiLayerAtmosphere
        The multi-layer atmospheric model corresponding to the site.
    '''
    if site == 'mauna_kea':
        layers = make_mauna_kea_atmospheric_layers(input_grid, cn_squared, outer_scale)
    elif site == 'las_campanas':
        layers = make_las_campanas_atmospheric_layers(input_grid, cn_squared, outer_scale)
    elif site == 'keck':
        layers = make_keck_atmospheric_layers(input_grid, cn_squared, outer_scale)
    else:
        raise NotImplementedError('Site unknown.')

    return MultiLayerAtmosphere(layers, **kwargs)
