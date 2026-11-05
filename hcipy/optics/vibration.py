import numpy as np

from .optical_element import OpticalElement
from ..mode_basis import ModeBasis
from ..util import StateSpaceDynamics
from .aberration import DynamicSurfaceAberration

class SimpleVibration(OpticalElement):
    '''A simple vibration model.

    This models a purely sinusoidal vibration of a single mode (such as tip or tilt).
    No noise is added: the motion is purely sinusoidal with a single frequency. However,
    the frequency and amplitude can be changed on the fly, if necessary.

    Parameters
    ----------
    mode : Field
        The spatial mode of the vibration.
    amplitude : scalar
        The amplitude of the vibration. After multiplying the mode and
        the amplitude, the resuilting units should be meters.
    frequency : scalar
        The temporal frequency of the vibration in Hz.
    phase_0 : scalar, optional
        The initial phase of the vibration in radians. Default is 0.

    Attributes
    ----------
    t : float
        The time in seconds.
    '''
    def __init__(self, mode, amplitude, frequency, phase_0=0):
        self.mode = mode
        self.amplitude = amplitude
        self.phase_0 = phase_0
        self._frequency = frequency
        self.t = 0

    @property
    def frequency(self):
        '''The temporal frequency of the vibration in Hz.
        '''
        return self._frequency

    @frequency.setter
    def frequency(self, frequency):
        delta_phase = 2 * np.pi * (self._frequency - frequency) * self.t
        self.phase_0 += delta_phase
        self._frequency = frequency

    @property
    def phase(self):
        '''The current phase of the vibration in radians.
        '''
        return 2 * np.pi * self.frequency * self.t + self.phase_0

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
        wf = wavefront.copy()

        wf.electric_field *= np.exp(1j * (self.mode * self.amplitude / wf.wavelength * np.sin(self.phase)))
        return wf

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
        wf = wavefront.copy()

        wf.electric_field *= np.exp(-1j * (self.mode * self.amplitude / wf.wavelength * np.sin(self.phase)))
        return wf

class DampedHarmonicVibration(DynamicSurfaceAberration):
    '''A damped harmonic oscillator vibration model driven by white noise.

    This class models a damped harmonic oscillator with natural frequency and
    damping ratio, driven by white noise with a given power spectral density.
    The model uses a continuous-time state space representation with exact
    covariance evolution.

    Parameters
    ----------
    mode : Field
        The spatial mode of the vibration. This mode is assumed to be normalized
        to an RMS of one.
    natural_frequency : scalar
        The natural frequency of the oscillator in Hz. Must be positive.
    damping_ratio : scalar
        The damping ratio (dimensionless). Must be positive.
    driving_psd : scalar
        The power spectral density of the driving acceleration noise in m^2 s^-4 per rad/s.
        This represents the strength of the external disturbance. Must be non-negative.
    refractive_index : float or callable, optional
        Refractive index of the medium. Can be a constant (float) or a callable
        that takes wavelength and returns refractive index. Default is 1.0.
    seed : int, optional
        Random seed for reproducibility. Default is None.
    '''
    def __init__(self, mode, natural_frequency, damping_ratio, driving_psd, refractive_index=1.0, seed=None):
        if driving_psd < 0:
            raise ValueError(f"driving_psd must be non-negative, got {driving_psd}.")

        if damping_ratio <= 0:
            raise ValueError(f"The damping ratio must be positive, got {damping_ratio}.")

        if natural_frequency <= 0:
            raise ValueError(f"The natural frequency must be positive, got {natural_frequency}.")

        self._omega_0 = 2 * np.pi * natural_frequency
        self._damping_ratio = damping_ratio
        self._driving_psd = driving_psd

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
        super().__init__(ModeBasis([mode]), dynamics, refractive_index)

    @property
    def damping_ratio(self):
        '''The damping ratio of the oscillator.
        '''
        return self._damping_ratio

    @property
    def driving_psd(self):
        '''The power spectral density of the driving white noise.
        '''
        return self._driving_psd

    @property
    def mode(self):
        '''The spatial mode of the vibration.
        '''
        return self.modes[0]

    @property
    def natural_frequency(self):
        '''The natural resonant frequency in Hz.
        '''
        return self._omega_0 / (2 * np.pi)

    @property
    def oscillation_frequency(self):
        '''The frequency of the damped oscillation in Hz.

        For critically damped (damping_ratio = 1) and overdamped
        (damping_ratio > 1) systems, the oscillator does not oscillate
        and this returns NaN.
        '''
        if self.damping_ratio >= 1:
            return np.nan

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
    def displacement(self):
        '''The current displacement of the oscillator in meters.

        Returns
        -------
        float
            The displacement in meters.
        '''
        return self.coefficients[0]
