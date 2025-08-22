from .atmospheric_model import Cn_squared_from_fried_parameter, MultiLayerAtmosphere
from .infinite_atmospheric_layer import InfiniteAtmosphericLayer
from ..dev import deprecated_name_changed

import numpy as np
import warnings

def make_mauna_kea_atmospheric_layers(input_grid, cn_squared=None, outer_scale=None):
    '''Create a multi-layer atmosphere for the Mauna Kea observatory site.

    The layer parameters are taken from [Guyon2005]_, in turn derived
    from [Tokovinin2005]_ and are representative for Mauna Kea, Hawaii.

    .. [Guyon2005] Olivier Guyon, "Limits of Adaptive Optics for
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

    Notes
    -----
    This function is deprecated. Use :func:`make_mauna_kea_atmospheric_layers` instead,
    or, even better,

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

def make_las_campanas_atmospheric_layers(input_grid, cn_squared=None, outer_scale=None, r0=0.16, L0=25, wavelength=550e-9):
    '''Create a multi-layer atmosphere for the Las Campanas Observatory site.

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
    r0 : scalar
        The integrated Cn^2 value for the atmosphere. This parameter is deprecated and
        should not be used. Use `cn_squared` instead.
    L0 : scalar
        The outer scale of the atmosphere. This parameter is deprecated and
        should not be used. Use `outer_scale` instead.
    wavelength : scalar
        The wavelength in meters at which to calculate the Fried parameter (default: 550nm).
        This parameter is deprecated and should not be used. Use `cn_squared` instead.

    Returns
    -------
    list
        A list of turbulence layers.
    '''
    if cn_squared is None:
        cn_squared = Cn_squared_from_fried_parameter(0.16)

    if outer_scale is None:
        outer_scale = 25

    # Detect users using previous signatures and emit deprecation warnings.
    if r0 != 0.16 or L0 != 25 or wavelength != 550e-9:
        warnings.warn('The signature of this function has changed. The old signature is deprecated. Please use the new signature.', DeprecationWarning, stacklevel=2)

        cn_squared = Cn_squared_from_fried_parameter(r0, wavelength)
        outer_scale = L0

    if cn_squared > 1e-6:
        # While a cn_squared value this high is unlikely for an atmosphere. Most likely they
        # gave a fried parameter instead of a cn_squared value. But we cannot assume that, so
        # we will only give a warning and assume the given cn_squared value was correct.

        warnings.warn('The signature of this function has changed. You may have given a fried parameter instead of cn_squared value.', DeprecationWarning, stacklevel=2)

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
