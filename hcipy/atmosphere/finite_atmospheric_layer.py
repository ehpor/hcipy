from .atmospheric_model import AtmosphericLayer, power_spectral_density_von_karman, fried_parameter_from_Cn_squared
import numpy as np
from ..util import SpectralNoiseFactoryMultiscale

import copy

class FiniteAtmosphericLayer(AtmosphericLayer):
    '''An atmospheric layer simulating atmospheric turbulence.

    This atmospheric layer is finite. This means that it will wrap
    after translation by more than `oversampling` times the extent of
    the input grid.

    Parameters
    ----------
    input_grid : Grid
        The grid on which the incoming wavefront is defined.
    Cn_squared : scalar
        The integrated strength of the turbulence for this layer.
    L0 : scalar
        The outer scale for the atmospheric turbulence of this layer.
        The default is infinity.
    velocity : scalar or array_like
        The wind speed for this atmospheric layer. If this is a scalar,
        the wind will be along x. If this is a 2D array, then the values
        are interpreted as the wind speed along x and y. The default is
        zero.
    height : scalar
        The height of the atmospheric layer. By itself, this value has no
        influence, but it'll be used by the AtmosphericModel to perform
        inter-layer propagations.
    oversampling : scalar
        The amount of oversampling in the Fourier space. The atmospheric layer
        will wrap after translation by more than `oversampling` times the extent of
        the input grid. The default is 2.
    seed : None, int, array of ints, SeedSequence, BitGenerator, Generator
        A seed to initialize the spectral noise. If None, then fresh, unpredictable
        entry will be pulled from the OS. If an int or array of ints, then it will
        be passed to a numpy.SeedSequency to derive the initial BitGenerator state.
        If a BitGenerator or Generator are passed, these will be wrapped and used
        instead. Default: None.
    '''
    def __init__(self, input_grid, Cn_squared=None, L0=np.inf, velocity=0, height=0, oversampling=2, seed=None):
        self._noise = None
        self._achromatic_screen = None

        AtmosphericLayer.__init__(self, input_grid, Cn_squared, L0, velocity, height)

        self.oversampling = oversampling
        self.center = np.zeros(2)

        self._original_rng = np.random.default_rng(seed)

        self.reset()

    def reset(self, make_independent_realization=False):
        '''Reset the atmospheric layer to t=0.

        Parameters
        ----------
        make_independent_realization : boolean
            Whether to start an independent realization of the noise for the
            atmospheric layer or not. When this is False, the exact same phase
            screens will be generated as in the first run, as long as the Cn^2
            and outer scale are the same. This allows for testing of multiple
            runs of a control system with different control parameters. When
            this is True, an independent realization of the atmospheric layer
            will be generated. This is useful for Monte-Carlo-style computations.
            The default is False.
        '''
        if make_independent_realization:
            # Reset the original random generator to the current one. This
            # will essentially reset the randomness.
            self._original_rng = copy.copy(self.rng)
        else:
            # Make a copy of the original random generator. This copy will be
            # used as the source for all randomness.
            self.rng = copy.copy(self._original_rng)

        self.psd = power_spectral_density_von_karman(fried_parameter_from_Cn_squared(self.Cn_squared, 1), self.L0)

        self.noise_factory = SpectralNoiseFactoryMultiscale(self.psd, self.input_grid, self.oversampling)
        self._noise = self.noise_factory.make_random(self.rng)

        self._achromatic_screen = None

    @property
    def noise(self):
        '''The (unshifted) spectral noise of this layer.

        This property is not intended to be used by the user.
        '''
        if self._noise is None:
            self.reset()

        return self._noise

    @property
    def achromatic_screen(self):
        '''The phase of this layer for a wavelength of one.

        This property is not intended to be used by the user.
        '''
        if self._achromatic_screen is None:
            self._achromatic_screen = self.noise.shifted(self.center)()

        return self._achromatic_screen

    def phase_for(self, wavelength):
        '''Compute the phase at a certain wavelength.

        Parameters
        ----------
        wavelength : scalar
            The wavelength of the light for which to compute the phase screen.

        Returns
        -------
        Field
            The computed phase screen.
        '''
        return self.achromatic_screen / wavelength

    def evolve_until(self, t):
        '''Evolve the atmospheric layer until a certain time.

        Parameters
        ----------
        t : scalar
            The new time to evolve the phase screen to.
        '''
        self.center = self.velocity * t
        self._achromatic_screen = None

    @property
    def Cn_squared(self):  # noqa: N802
        '''The integrated strength of the turbulence for this layer.
        '''
        return self._Cn_squared

    @Cn_squared.setter
    def Cn_squared(self, Cn_squared):  # noqa: N802
        self._Cn_squared = Cn_squared
        self._noise = None

    @property
    def outer_scale(self):
        '''The outer scale of the turbulence for this layer.
        '''
        return self._L0

    @outer_scale.setter
    def L0(self, L0):  # noqa: N802
        self._L0 = L0
        self._noise = None
