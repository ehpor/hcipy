from .atmospheric_model import Cn_squared_from_fried_parameter, MultiLayerAtmosphere
from .infinite_atmospheric_layer import InfiniteAtmosphericLayer
from ..dev import deprecated_name_changed

import numpy as np

def make_mauna_kea_atmospheric_layers(input_grid, cn_squared=None, outer_scale=None):
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
    cn_squared : scalar
        The integrated Cn^2 value for all layers. If this is None (default),
        then a Fried parameter of 0.2m at 500nm is used.
    outer_scale : scalar or None
        The outer scale of the atmosphere. If this is None (default), then an
        outer scale of 10m is used.

    Returns
    -------
    list
        A list of turbulence layers.
    '''
    if cn_squared is None:
        cn_squared = Cn_squared_from_fried_parameter(0.2)

    if outer_scale is None:
        outer_scale = 10

    heights = np.array([500, 1000, 2000, 4000, 8000, 16000])
    velocities = np.array([10, 10, 10, 10, 10, 10])
    cn_squared_values = np.array([0.2283, 0.0883, 0.0666, 0.1458, 0.3350, 0.1350])

    cn_squared_values *= cn_squared / np.sum(cn_squared_values)

    layers = []
    for h, v, cn in zip(heights, velocities, cn_squared_values):
        layers.append(InfiniteAtmosphericLayer(input_grid, cn, outer_scale, v, h, 2))

    return layers

@deprecated_name_changed(make_mauna_kea_atmospheric_layers)
def make_standard_atmospheric_layers(input_grid, L0=10):
    '''Make a standard set of atmospheric layers.

    Parameters
    ----------
    input_grid : Grid
        The input grid for the atmospheric layers.
    L0 : scalar
        The outer scale of the atmosphere.

    Returns
    -------
    list
        A list of turbulence layers.
    '''
    return make_mauna_kea_atmospheric_layers(input_grid, outer_scale=L0)

def make_las_campanas_atmospheric_layers(input_grid, cn_squared=None, outer_scale=None):
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
    cn_squared : scalar
        The integrated Cn^2 value for all layers. If this is None (default),
        then a Fried parameter of 0.16m at 500nm is used.
    outer_scale : scalar or None
        The outer scale of the atmosphere. If this is None (default), then an
        outer scale of 25m is used.

    Returns
    -------
    list
        A list of turbulence layers.
    '''
    if cn_squared is None:
        cn_squared = Cn_squared_from_fried_parameter(0.16)

    if outer_scale is None:
        outer_scale = 25

    heights = np.array([250, 500, 1000, 2000, 4000, 8000, 16000])
    velocities = np.array([10, 10, 20, 20, 25, 30, 25])

    cn_squared_values = np.array([0.42, 0.03, 0.06, 0.16, 0.11, 0.10, 0.12])

    cn_squared_values *= cn_squared / np.sum(cn_squared_values)

    layers = []
    for h, v, cn in zip(heights, velocities, cn_squared_values):
        layers.append(InfiniteAtmosphericLayer(input_grid, cn, outer_scale, v, h, 2))

    return layers

def make_keck_atmospheric_layers(input_grid, cn_squared=None, outer_scale=20):
    '''Creates a multi-layer atmosphere for Keck Observatory.

    The atmospheric parameters are based off of [Keck AO note 303]_. The default
    Fried parameter is 0.2m and the default outer scale is 20m.

    .. [Keck AO note 303] https://www2.keck.hawaii.edu/optics/kpao/files/KAON/KAON303.pdf

    Parameters
    ----------
    input_grid : Grid
        The input grid of the atmospheric layers.
    cn_squared : scalar
        The integrated Cn^2 value for all layers. If this is None (default),
        then a Fried parameter of 0.2m at 500nm is used.
    outer_scale : scalar or None
        The outer scale of the atmosphere. If this is None (default), then an
        outer scale of 20m is used.

    Returns
    -------
    list
        A list of turbulence layers.
    '''
    if cn_squared is None:
        cn_squared = Cn_squared_from_fried_parameter(0.2)

    if outer_scale is None:
        outer_scale = 20

    heights = np.array([0, 2100, 4100, 6500, 9000, 12000, 14800])
    velocities = np.array([6.7, 13.9, 20.8, 29.0, 29.0, 29.0, 29.0])
    cn_squared_values = np.array([0.369, 0.219, 0.127, 0.101, 0.046, 0.111, 0.027])

    cn_squared_values *= cn_squared / np.sum(cn_squared_values)

    layers = []
    for h, v, cn in zip(heights, velocities, cn_squared_values):
        layers.append(InfiniteAtmosphericLayer(input_grid, cn, outer_scale, v, h, 2))

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

    Raises
    ------
    ValueError
        If the requested `site` is not one of the implemented sites.
    '''
    if site == 'mauna_kea':
        layers = make_mauna_kea_atmospheric_layers(input_grid, cn_squared, outer_scale)
    elif site == 'las_campanas':
        layers = make_las_campanas_atmospheric_layers(input_grid, cn_squared, outer_scale)
    elif site == 'keck':
        layers = make_keck_atmospheric_layers(input_grid, cn_squared, outer_scale)
    else:
        raise ValueError('Site unknown.')

    return MultiLayerAtmosphere(layers, **kwargs)
