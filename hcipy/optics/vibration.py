import numpy as np
from scipy.linalg import expm, solve_continuous_lyapunov

from .optical_element import OpticalElement
from ..mode_basis import ModeBasis
from ..util import StateSpaceDynamics

class SimpleVibration(OpticalElement):
    def __init__(self, mode, amplitude, frequency, phase_0=0):
        self.mode = mode
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase_0 = phase_0
        self.t = 0

    @property
    def frequency(self):
        return self._frequency

    @frequency.setter
    def frequency(self, frequency):
        delta_phase = 2 * np.pi * (self.frequency + self.frequency)
        self.phase_0 += delta_phase
        self._frequency = frequency

    @property
    def phase(self):
        return 2 * np.pi * self.frequency * self.t + self.phase_0

    def forward(self, wavefront):
        wf = wavefront.copy()

        wf.electric_field *= np.exp(1j * (self.mode * self.amplitude / wf.wavelength * np.sin(self.phase)))
        return wf

    def backward(self, wavefront):
        wf = wavefront.copy()

        wf.electric_field *= np.exp(-1j * (self.mode * self.amplitude / wf.wavelength * np.sin(self.phase)))
        return wf

class DampedHarmonicVibration(OpticalElement):
    '''A damped harmonic oscillator vibration model driven by white noise.

    This class models a damped harmonic oscillator with natural frequency and
    damping ratio, driven by white noise with a given power spectral density.
    The model uses a continuous-time state space representation with exact
    covariance evolution.

    This class uses composition with :class:`DynamicSurfaceAberration` to
    apply the time-varying phase modulation.

    Parameters
    ----------
    mode : Field
        The spatial mode of the vibration. This mode is assumed to be normalized
        to an RMS of one.
    natural_frequency : scalar
        The natural frequency of the oscillator in Hz.
    damping_ratio : scalar
        The damping ratio (dimensionless). Must be positive.
    driving_psd : scalar
        The power spectral density of the driving white noise in m^2/Hz. This
        represents the strength of the external disturbance. Must be non-negative.
    refractive_index : float or callable, optional
        Refractive index of the medium. Can be a constant (float) or a callable
        that takes wavelength and returns refractive index. Default is 1.0.
    seed : int, optional
        Random seed for reproducibility. Default is None.
    '''
    def __init__(self, mode, natural_frequency, damping_ratio, driving_psd, refractive_index=1.0, seed=None):
        if driving_psd <= 0:
            raise ValueError(f"driving_psd must be non-negative, got {driving_psd}.")

        if damping_ratio <= 0:
            raise ValueError(f"The damping ratio must be positive, got {damping_ratio}.")

        self._omega_0 = 2 * np.pi * natural_frequency
        self.damping_ratio = damping_ratio
        self.driving_psd = driving_psd

        # Create state space dynamics for damped harmonic oscillator
        # Continuous-time state matrix
        A = np.array([[0, 1], [-self._omega_0**2, -2 * self.damping_ratio * self._omega_0]])

        # Input matrix (noise affects velocity)
        B = np.array([[0], [np.sqrt(driving_psd)]])

        # Observation matrix (we observe position/displacement)
        C = np.array([[1, 0]])

        # Create state space dynamics with random seed
        dynamics = StateSpaceDynamics(A, B, C, seed=seed)

        # Create mode basis from single mode
        modes = ModeBasis([mode])

        # Create dynamic surface aberration
        self._aberration = DynamicSurfaceAberration(modes, dynamics, refractive_index)

    @property
    def mode(self):
        '''The spatial mode of the vibration.
        '''
        return self._aberration.modes[0]

    @property
    def natural_frequency(self):
        '''The natural resonant frequency in Hz.
        '''
        return self._omega_0 / (2 * np.pi)

    @property
    def oscillation_frequency(self):
        '''The oscillation frequency in Hz.
        '''
        return self.natural_frequency * np.sqrt(1 - self.damping_ratio**2)

    @property
    def quality_factor(self):
        '''The quality factor of the oscillator.
        '''
        return 1 / (2 * self.damping_ratio)

    @property
    def rms_displacement(self):
        '''The stationary RMS displacement in meters.

        For a damped harmonic oscillator driven by white noise with PSD S_0,
        the stationary RMS displacement is sqrt(S_0 / (4*zeta*omega_0^3)).

        Returns
        -------
        float
            The stationary RMS displacement in meters.
        '''
        return np.sqrt(self.driving_psd / (4 * self.damping_ratio * self._omega_0**3))

    @property
    def t(self):
        '''The current time in seconds.
        '''
        return self._aberration.t

    @t.setter
    def t(self, t):
        self._aberration.t = t

    @property
    def displacement(self):
        '''The current displacement of the oscillator in meters.

        Returns
        -------
        float
            The displacement in meters.
        '''
        return self._aberration.coefficients[0]

    def evolve_until(self, t):
        '''Evolve the oscillator until time t.

        Parameters
        ----------
        t : scalar
            The target time in seconds. Must be >= current time.
        '''
        self._aberration.evolve_until(t)

    def forward(self, wavefront):
        '''Propagate the wavefront forward through the vibration.

        Parameters
        ----------
        wavefront : Wavefront
            The input wavefront.

        Returns
        -------
        Wavefront
            The output wavefront with the vibration applied.
        '''
        return self._aberration.forward(wavefront)

    def backward(self, wavefront):
        '''Propagate the wavefront backward through the vibration.

        Parameters
        ----------
        wavefront : Wavefront
            The input wavefront.

        Returns
        -------
        Wavefront
            The output wavefront with the inverse vibration applied.
        '''
        return self._aberration.backward(wavefront)

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
