import numpy as np
from scipy.linalg import expm, solve_continuous_lyapunov

from .optical_element import OpticalElement

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

class DampedOscillatorVibration(OpticalElement):
    '''A damped harmonic oscillator vibration model driven by white noise.

    This class models a damped harmonic oscillator with natural frequency and
    damping ratio, driven by white noise with a given power spectral density.
    The model uses an autoregressive (AR(2)) process with exact analytical coefficients
    computed on the fly for any timestep.

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
    seed : int, optional
        Random seed for reproducibility. Default is None.
    '''
    def __init__(self, mode, natural_frequency, damping_ratio, driving_psd, seed=None):
        self.mode = mode
        self._omega_0 = 2 * np.pi * natural_frequency
        self.damping_ratio = damping_ratio
        self.driving_psd = driving_psd

        self._t = 0
        self.rng = np.random.default_rng(seed)

        if driving_psd <= 0:
            raise ValueError(f"driving_psd must be non-negative, got {driving_psd}.")

        if damping_ratio <= 0:
            raise ValueError(f"The damping ratio must be positive, got {damping_ratio}.")

        # Initialize state for position and velocity from stationary distribution
        var_x = self.driving_psd / (4 * self.damping_ratio * self._omega_0**3)
        var_v = self.driving_psd * self._omega_0 / (4 * self.damping_ratio)

        self._state = np.array([
            self.rng.standard_normal() * np.sqrt(var_x),
            self.rng.standard_normal() * np.sqrt(var_v)
        ])

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
        return self._t

    @t.setter
    def t(self, t):
        self.evolve_until(t)

    @property
    def displacement(self):
        '''The current displacement of the oscillator in meters.

        Returns
        -------
        float
            The displacement in meters.
        '''
        return self._state[0]

    def _compute_state_transition_matrix(self, dt):
        '''Compute state transition matrix for a given timestep.

        For a damped harmonic oscillator, the state vector is [x, v] where x is position
        and v is velocity. The state transition matrix A is derived from the continuous-time
        dynamics: dx/dt = v, dv/dt = -ω₀²x - 2ζω₀v.

        The exact solution for free evolution (no forcing) is:
        x(t+dt) = a*x(t) + b*v(t)
        v(t+dt) = c*x(t) + d*v(t)

        Parameters
        ----------
        dt : scalar
            The timestep in seconds.

        Returns
        -------
        A : ndarray
            The 2x2 state transition matrix [[a, b], [c, d]].
        '''
        omega_0 = self._omega_0
        zeta = self.damping_ratio

        omega_d = omega_0 * np.sqrt(1 - zeta**2)
        decay = np.exp(-zeta * omega_0 * dt)

        cos_wdt = np.cos(omega_d * dt)
        sin_wdt = np.sin(omega_d * dt)

        # State transition matrix elements
        a = decay * (cos_wdt + zeta * omega_0 / omega_d * sin_wdt)
        b = decay * sin_wdt / omega_d
        c = -omega_0**2 * decay * sin_wdt / omega_d
        d = decay * (cos_wdt - zeta * omega_0 / omega_d * sin_wdt)

        return np.array([[a, b], [c, d]])

    def evolve_until(self, t):
        '''Evolve the oscillator until time t.

        This method computes the state transition on the fly for the exact
        timestep Δt = t - current_time, allowing arbitrary time propagation.

        Parameters
        ----------
        t : scalar
            The target time in seconds. Must be >= current time.

        Raises
        ------
        ValueError
            If t is less than the current time (backward evolution not allowed).
        '''
        if t < self._t:
            raise ValueError(f"Cannot evolve backward in time. Current time is {self._t}, requested time is {t}.")

        delta_t = t - self._t

        if delta_t == 0:
            return

        omega_0 = self._omega_0
        zeta = self.damping_ratio

        # Continuous-time state matrix
        A_cont = np.array([[0, 1], [-omega_0**2, -2*zeta*omega_0]])

        # Input matrix (noise affects velocity)
        B = np.array([[0], [1]])

        # Noise intensity
        Q_cont = B @ np.array([[self.driving_psd]]) @ B.T

        # Solve continuous Lyapunov equation for stationary covariance
        P = solve_continuous_lyapunov(A_cont, -Q_cont)

        # Discrete-time state transition matrix
        A_disc = expm(A_cont * delta_t)

        # Discrete-time noise covariance from discrete Lyapunov equation
        Q_disc = P - A_disc @ P @ A_disc.T

        # Ensure positive semi-definite by symmetrizing
        Q_disc = (Q_disc + Q_disc.T) / 2

        # Generate noise vector from multivariate normal with covariance Q_disc
        noise = self.rng.multivariate_normal([0, 0], Q_disc)

        # Evolve state
        self._state = A_disc @ self._state + noise
        self._t = t

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

        phase_shift = self.mode * self.displacement / wf.wavelength * 2 * np.pi
        wf.electric_field *= np.exp(1j * phase_shift)
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

        phase_shift = self.mode * self.displacement / wf.wavelength * 2 * np.pi
        wf.electric_field *= np.exp(-1j * phase_shift)
        return wf
