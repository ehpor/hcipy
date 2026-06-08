import numpy as np

from ..field import Field
from ..util import SpectralNoiseFactoryFFT, inverse_tikhonov
from .apodization import SurfaceApodizer
from ..propagation import FresnelPropagator
from ..aperture import make_circular_aperture, make_super_gaussian_aperture
from .optical_element import OpticalElement
from ..fourier import FourierFilter

def make_power_law_error(pupil_grid, ptv, diameter, exponent=-2.5, aperture=None, remove_modes=None):
    '''Create an error surface from a power-law power spectral density.

    Parameters
    ----------
    pupil_grid : Grid
        The grid on which to calculate the error.
    ptv : scalar
        The peak-to-valley of the wavefront aberration in meters.
    diameter : scalar
        The diameter over which the ptv is calculated.
    exponent : scalar
        The exponent of the power law.
    aperture : Field
        The mask over which to calculate the ptv. A circular aperture with diameter
        `diameter` is used if this is not given.
    remove_modes : ModeBasis
        The modes which to remove from the surface aberration. The peak-to-valley
        is enforced before these modes are removed. This allows for correctting surface
        errors with optic alignment.

    Returns
    -------
    Field
        The surface error calculated on `pupil_grid`.
    '''
    def psd(grid):
        res = Field(grid.as_('polar').r**exponent, grid)
        res[grid.as_('polar').r == 0] = 0
        return res

    if aperture is None:
        aperture = make_circular_aperture(diameter)(pupil_grid)

    screen = SpectralNoiseFactoryFFT(psd, pupil_grid).make_random()()
    screen *= ptv / np.ptp(screen[aperture != 0])

    if remove_modes is not None:
        trans = remove_modes.transformation_matrix
        trans_inv = inverse_tikhonov(trans, 1e-6)
        screen -= trans.dot(trans_inv.dot(screen))

    return Field(screen * aperture, pupil_grid)

def make_high_pass_power_law_error(pupil_grid, ptv, diameter, cutoff_frequency, exponent=-2.5, aperture=None, filter_shape_parameter=15):
    '''Create an error surface from a high-pass filtered power-law power spectral density.

    Parameters
    ----------
    pupil_grid : Grid
        The grid on which to calculate the error.
    ptv : scalar
        The peak-to-valley of the wavefront aberration in meters before filtering.
    diameter : scalar
        The diameter over which the ptv is calculated.
    cutoff_frequency : scalar
        The cutoff frequency of the high-pass filter.
    exponent : scalar
        The exponent of the power law.
    aperture : Field
        The mask over which to calculate the ptv. A circular aperture with diameter
        `diameter` is used if this is not given.
    filter_shape_parameter : scalar
        The shape parameter for the super Gaussian high-pass filter fuction. Default value is 15.0

    Returns
    -------
    Field
        The surface error calculated on `pupil_grid`.
    '''
    def filter_function(fourier_grid):
        return 1 - make_super_gaussian_aperture(2 * cutoff_frequency, filter_shape_parameter)(fourier_grid)

    ff = FourierFilter(pupil_grid, filter_function, q=2)

    if aperture is None:
        aperture = make_circular_aperture(diameter)(pupil_grid)
    aperture_mask = (aperture > 0).astype(float)

    phase_screen = make_power_law_error(pupil_grid, ptv, diameter, exponent=exponent, aperture=None)
    phase_screen = ff.forward(phase_screen).real

    return phase_screen * aperture_mask

class SurfaceAberration(SurfaceApodizer):
    '''A surface aberration with a specific power law.

    Parameters
    ----------
    pupil_grid : Grid
        The grid on which the incoming wavefront is defined.
    ptv : scalar
        The peak-to-valley of the wavefront aberration in meters.
    diameter : scalar
        The diameter over which the ptv is calculated.
    exponent : scalar
        The exponent of the power law.
    refractive_index : scalar
        The refractive index of the surface for which this is the surface error.
        The default is a mirror surface.
    aperture : Field
        The mask over which to calculate the ptv. A circular aperture with diameter
        `diameter` is used if this is not given.
    remove_modes : ModeBasis
        The modes which to remove from the surface aberration. The peak-to-valley
        is enforced before these modes are removed. This allows for correcting surface
        errors with optic alignment.
    '''
    def __init__(self, pupil_grid, ptv, diameter, exponent=-2.5, refractive_index=-1, aperture=None, remove_modes=None):
        surface = make_power_law_error(pupil_grid, ptv, diameter, exponent, aperture, remove_modes)
        SurfaceApodizer.__init__(self, surface, refractive_index)

class SurfaceAberrationAtDistance(OpticalElement):
    '''A surface at a certain distance from the current plane.

    Light is propagated to this surface, then the surface errors are
    applied, and afterwards the light is propagated back towards the
    original plane. This allows for easy addition of surface errors on
    lenses, while still retaining the Fraunhofer propagations in between
    focal and pupil planes.

    Parameters
    ----------
    surface_aberration : OpticalElement
        The optical element describing the surface aberration.
    distance : scalar
        The distance from the current plane.
    '''
    def __init__(self, surface_aberration, distance):
        self.fresnel = FresnelPropagator(surface_aberration.input_grid, distance)
        self.surface_aberration = surface_aberration

    def forward(self, wavefront):
        '''Propagate a wavefront forwards through the surface aberration.

        Parameters
        ----------
        wavefront : Wavefront
            The incoming wavefront.

        Returns
        -------
        Wavefront
            The wavefront after the surface aberration. This wavefront is
            given at the same plane as `wavefront`.
        '''
        wf = self.fresnel.forward(wavefront)
        wf = self.surface_aberration.forward(wf)
        return self.fresnel.backward(wf)

    def backward(self, wavefront):
        '''Propagate a wavefront backwards through the surface aberration.

        Parameters
        ----------
        wavefront : Wavefront
            The incoming wavefront.

        Returns
        -------
        Wavefront
            The wavefront before the surface aberration. This wavefront is
            given at the same plane as `wavefront`.
        '''
        wf = self.fresnel.forward(wavefront)
        wf = self.surface_aberration.backward(wf)
        return self.fresnel.backward(wf)

class DynamicSurfaceAberration(OpticalElement):
    """A dynamic surface aberration using state space dynamics.

    This class models time-varying surface aberrations using a continuous-time
    state space model. It wraps a :class:`StateSpaceDynamics` object and applies
    phase modulation to wavefronts based on the current state.

    Parameters
    ----------
    modes : ModeBasis
        Spatial modes defining the surface aberration pattern. These represent
        surface height in meters.
    dynamics : StateSpaceDynamics
        The state space dynamics object that provides time-varying coefficients.
    refractive_index : float or callable, optional
        Refractive index of the medium. Can be a constant (float) or a callable
        that takes wavelength and returns refractive index. Default is 1.0.

    Attributes
    ----------
    modes : ModeBasis
        Spatial modes defining the surface aberration pattern.
    dynamics : StateSpaceDynamics
        The underlying state space dynamics object.
    refractive_index : float or callable
        The refractive index.
    """
    def __init__(self, modes, dynamics, refractive_index=1.0):
        if len(modes) != dynamics.num_outputs:
            raise ValueError(f"Number of modes ({len(modes)}) must match number of dynamics outputs ({dynamics.num_outputs})")

        self.modes = modes
        self.dynamics = dynamics
        self.refractive_index = refractive_index

    @property
    def state(self):
        """Current internal state vector.
        """
        return self.dynamics.state

    @property
    def coefficients(self):
        """Current mode coefficients (y = C @ x).
        """
        return self.dynamics.coefficients

    @property
    def t(self):
        """Current simulation time.
        """
        return self.dynamics.t

    @t.setter
    def t(self, t):
        self.evolve_until(t)

    def evolve_until(self, t):
        """Evolve the state from current time to time t.

        Parameters
        ----------
        t : float
            Target time to evolve to.
        """
        self.dynamics.evolve_until(t)

    @property
    def surface(self):
        """The surface """
        return self.modes.linear_combination(self.coefficients)

    def _get_phase_multiplier(self, wavelength):
        """Get the phase multiplier based on our refractive index.
        """
        if callable(self.refractive_index):
            n = self.refractive_index(wavelength)
        else:
            n = self.refractive_index
        return 2 * np.pi * (n - 1) / wavelength

    def forward(self, wavefront):
        """Propagate wavefront forward through the aberration.

        Applies phase modulation based on the current state and modes.

        Parameters
        ----------
        wavefront : Wavefront
            Input wavefront.

        Returns
        -------
        Wavefront
            Modulated wavefront with applied phase.
        """
        wf = wavefront.copy()

        phase_multiplier = self._get_phase_multiplier(wf.wavelength)
        wf.electric_field *= np.exp(1j * phase_multiplier * self.surface)

        return wf

    def backward(self, wavefront):
        """Propagate wavefront backward through the aberration.

        Applies inverse phase modulation based on the current state and modes.

        Parameters
        ----------
        wavefront : Wavefront
            Input wavefront.

        Returns
        -------
        Wavefront
            Modulated wavefront with inverse phase.
        """
        wf = wavefront.copy()

        phase_multiplier = self._get_phase_multiplier(wf.wavelength)
        wf.electric_field *= np.exp(-1j * phase_multiplier * self.surface)

        return wf
