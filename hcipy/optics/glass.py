import numpy as np

try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files

def make_sellmeier_glass(A, K, L):
    r'''The Sellmeier equation for the dispersion of the refractive index of materials.

    The dispersion relation of the Sellmeier equation is [Sellmeier1872]_,

    .. math:: n^2 = 1 + \sum_{i}{\frac{K_i \lambda^2}{\lambda^2-L_i}}

    .. [Sellmeier1872] W. Sellmeier 1872, Ueber die durch die Aetherschwingungen erregten
        Mitschwingungen der Körpertheilchen und deren Rückwirkung auf die ersteren, besonders
        zur Erklärung der Dispersion und ihrer Anomalien (II. Theil),"  Annalen der Physik und
        Chemie. 223 (11), 386–403 (1872)

    Parameters
    ----------
    A : scalar
        The Sellmeier coefficient
    K : array_like
        The Sellmeier coefficients
    L : array_like
        The Sellmeier coefficients in um**2

    Returns
    -------
    function
        The refractive index profile function
    '''
    def refractive_index(wavelength):
        n_squared = A
        for Ki, Li in zip(K, L):
            n_squared += Ki * (wavelength * 1e6)**2 / ((wavelength * 1e6)**2 - Li)
        return np.sqrt(n_squared)

    return refractive_index

def make_cauchy_glass(coefficients):
    r'''The Cauchy equation for the dispersion of the refractive index of materials.

    The dispersion relation of the Cauchy equation is,

    .. math:: n = \sum_{i=0}{\frac{K_i}{\lambda^{2i}}

    Parameters
    ----------
    coefficients : scalar or array_like
        The Cauchy coefficients in um**(2*i).

    Returns
    -------
    function
        The refractive index profile function
    '''
    def refractive_index(wavelength):
        n = 0
        for i, coef in enumerate(coefficients):
            n += coef / (wavelength * 1e6)**(2 * i)
        return n

    return refractive_index

_glass_catalogue = None

def get_refractive_index(glass_name):
    '''Get the refractive index for one of the glasses in the catalogue.

    Parameters
    ----------
    glass_name : str
        The name of the glass.

    Returns
    -------
    callable
        The refractive index as function of wavelength.
    '''
    if _glass_catalogue is None:
        _build_glass_catalogue()

    if glass_name not in _glass_catalogue:
        # Not found. Try to find suggestion.
        for key in _glass_catalogue.keys():
            if key.lower() == glass_name.lower():
                raise ValueError('The requested glass "%s" was not found in the catalogue. Did you mean "%s"?' % (glass_name, key))

        # No suggestion found, raise
        raise ValueError('The requested glass "%s" was not found in the catalogue.' % glass_name)

    return _glass_catalogue[glass_name]

def get_glasses_in_catalogue():
    '''Get the names of all the glasses in the glass catalogue.

    Returns
    -------
    iterable of strings
        The names of each of the glasses in the catalogue in arbitrary order.
    '''
    if _glass_catalogue is None:
        _build_glass_catalogue()

    return _glass_catalogue.keys()

def _build_glass_catalogue():
    global _glass_catalogue

    _glass_catalogue = {
        'IP_DIP': make_cauchy_glass([1.5273, 6.5456E-3, 2.5345E-4]),
        'SILICA': make_sellmeier_glass(1, [0.6961663, 0.4079426, 0.8974794], [0.0684043**2, 0.1162414**2, 9.896161**2]),
        'SAPPHIRE': make_sellmeier_glass(1, [1.4313493, 0.65054713, 5.3414021], [0.0726631**2, 0.1193242**2, 18.028251**2]),
        'CAF2': make_sellmeier_glass(1.33973, [0.69913, 0.11994, 4.35181], [0.09374**2, 21.18**2, 38.46**2]),
        'N-PSK53': make_sellmeier_glass(1.0, [1.3434087, 0.241417935, 0.952896897], [0.00675074317, 0.0219910513, 103.551457]),
        'VACUUM': lambda wavelengths: np.ones_like(wavelengths)
    }

    catalogues = ['schott_glass_catalogue_2018_09.csv', 'ohara_glass_catalogue_2019_08.csv']

    for fname in catalogues:
        catalogue = files('hcipy.optics').joinpath(fname)
        with catalogue.open() as f:
            _glass_catalogue.update(_parse_sellmeier_glass_catalogue(f))

def _parse_sellmeier_glass_catalogue(catalogue_file):
    data = np.loadtxt(catalogue_file, dtype=str, delimiter=',', skiprows=1)

    database = {}
    for di in data:
        database[di[0].replace(" ", "")] = make_sellmeier_glass(1, di[1:4].astype('float'), di[4::].astype('float'))

    return database
