import warnings

import numpy as np

__all__ = [
    'RefractiveIndexMaterial'
]

# The d, F and C spectral lines in meters.
_LINE_D = 0.5875618e-6
_LINE_F = 0.4861327e-6
_LINE_C = 0.6562725e-6

class RefractiveIndexMaterial:
    '''A material from the refractiveindex.info database.

    The refractive index is given by n + 1j*k, with the complex refractive
    index returned by calling the material. Wavelengths are in meters.

    Parameters
    ----------
    page : str
        The name of the page in the database.
    n_function : function or None
        A function of the wavelength (in meters) returning the refractive
        index. None if the material has no refractive index data.
    k_function : function or None, optional
        A function of the wavelength (in meters) returning the extinction
        coefficient. Defaults to None, in which case the extinction
        coefficient is assumed to be zero.
    shelf : str, optional
        The shelf in the refractiveindex.info database.
    book : str, optional
        The book in the refractiveindex.info database.
    wavelength_range : (float, float) or None, optional
        The valid wavelength range of the material in meters. Evaluating the
        material outside this range produces a warning, unless
        allow_extrapolation is set.
    '''
    def __init__(self, page, n_function, k_function=None, *, shelf=None, book=None, wavelength_range=None):
        self.page = page
        self.n_function = n_function
        self.k_function = k_function
        self.shelf = shelf
        self.book = book
        self.wavelength_range = wavelength_range

    def _check_wavelength_range(self, wavelength, allow_extrapolation):
        if allow_extrapolation or self.wavelength_range is None:
            return

        lo, hi = self.wavelength_range
        wl = np.asarray(wavelength)

        if np.any(wl < lo) or np.any(wl > hi):
            warnings.warn(
                f'Material {self.page!r} evaluated outside its valid wavelength range '
                f'[{lo * 1e6:.3g}, {hi * 1e6:.3g}] um.',
                UserWarning,
                stacklevel=3)

    def __call__(self, wavelength, allow_extrapolation=False):
        '''Get the complex refractive index n + 1j*k at the given wavelength.

        Parameters
        ----------
        wavelength : scalar or array_like
            The wavelength in meters.
        allow_extrapolation : bool, optional
            Allow evaluation outside the valid wavelength range without
            a warning. Defaults to False.

        Returns
        -------
        scalar or array_like
            The complex refractive index.
        '''
        return self.n(wavelength, allow_extrapolation) + 1j * self.k(wavelength, allow_extrapolation)

    def n(self, wavelength, allow_extrapolation=False):
        '''Get the real refractive index n at the given wavelength.

        Parameters
        ----------
        wavelength : scalar or array_like
            The wavelength in meters.
        allow_extrapolation : bool, optional
            Allow evaluation outside the valid wavelength range without
            a warning. Defaults to False.

        Returns
        -------
        scalar or array_like
            The real refractive index.
        '''
        if self.n_function is None:
            raise ValueError(f'Material {self.page!r} has no refractive index data.')
        self._check_wavelength_range(wavelength, allow_extrapolation)
        return self.n_function(wavelength)

    def k(self, wavelength, allow_extrapolation=False):
        '''Get the extinction coefficient k at the given wavelength.

        Parameters
        ----------
        wavelength : scalar or array_like
            The wavelength in meters.
        allow_extrapolation : bool, optional
            Allow evaluation outside the valid wavelength range without
            a warning. Defaults to False.

        Returns
        -------
        scalar or array_like
            The extinction coefficient. Zero if no extinction data is
            available for this material.
        '''
        if self.k_function is None:
            return wavelength * 0
        self._check_wavelength_range(wavelength, allow_extrapolation)
        return self.k_function(wavelength)

    def dispersion(self, wavelength_1, wavelength_2):
        '''Get the difference in refractive index n(wavelength_1) - n(wavelength_2).

        Parameters
        ----------
        wavelength_1 : scalar or array_like
            The first wavelength in meters.
        wavelength_2 : scalar or array_like
            The second wavelength in meters.

        Returns
        -------
        scalar or array_like
            The difference in refractive index.
        '''
        return self.n(wavelength_1) - self.n(wavelength_2)

    def partial_dispersion(self, wavelength_1, wavelength_2, wavelength_3, wavelength_4):
        '''Get the partial dispersion ratio.

        The partial dispersion is defined as
        (n(wavelength_1) - n(wavelength_2)) / (n(wavelength_3) - n(wavelength_4)).

        Parameters
        ----------
        wavelength_1 : scalar or array_like
            The first wavelength in meters.
        wavelength_2 : scalar or array_like
            The second wavelength in meters.
        wavelength_3 : scalar or array_like
            The third wavelength in meters.
        wavelength_4 : scalar or array_like
            The fourth wavelength in meters.

        Returns
        -------
        scalar or array_like
            The partial dispersion ratio.
        '''
        return self.dispersion(wavelength_1, wavelength_2) / self.dispersion(wavelength_3, wavelength_4)

    def abbe(self, wavelength_short=None, wavelength_center=None, wavelength_long=None):
        '''Get the Abbe number.

        The Abbe number is defined as
        (n(wavelength_center) - 1) / (n(wavelength_short) - n(wavelength_long)).
        By default, the F, d and C spectral lines are used.

        Parameters
        ----------
        wavelength_short : scalar, optional
            The short wavelength in meters. Defaults to the F spectral line.
        wavelength_center : scalar, optional
            The center wavelength in meters. Defaults to the d spectral line.
        wavelength_long : scalar, optional
            The long wavelength in meters. Defaults to the C spectral line.

        Returns
        -------
        scalar
            The Abbe number.
        '''
        if wavelength_short is None:
            wavelength_short = _LINE_F
        if wavelength_center is None:
            wavelength_center = _LINE_D
        if wavelength_long is None:
            wavelength_long = _LINE_C

        n_center = self.n(wavelength_center)
        n_short = self.n(wavelength_short)
        n_long = self.n(wavelength_long)

        return (n_center - 1) / (n_short - n_long)

    @property
    def page_info(self):
        '''Get the provenance of this material.

        Returns
        -------
        dict
            A dictionary with the keys 'shelf', 'book', 'page' and
            'wavelength_range'.
        '''
        return {
            'shelf': self.shelf,
            'book': self.book,
            'page': self.page,
            'wavelength_range': self.wavelength_range
        }

    def __repr__(self):
        return f'RefractiveIndexMaterial({self.page!r}, shelf={self.shelf!r}, book={self.book!r})'
