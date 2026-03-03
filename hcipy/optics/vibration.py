import numpy as np

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

        # Initialize state for AR(2) process with samples from stationary distribution
        var_target = self.driving_psd / (4 * self.damping_ratio * self._omega_0**3)
        self._state = self.rng.standard_normal(size=2) * np.sqrt(var_target)

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

    def _compute_ar_coefficients(self, dt):
        '''Compute AR(2) coefficients for a given timestep.

        For a damped harmonic oscillator driven by white noise, we can analytically
        compute the AR(2) coefficients for any timestep Δt from the continuous-time
        parameters (ω₀, ζ).

        Parameters
        ----------
        dt : scalar
            The timestep in seconds.

        Returns
        -------
        phi : ndarray
            The AR coefficients [φ₁, φ₂].
        noise_std : scalar
            The standard deviation of the driving noise for this timestep.
        '''
        omega_d = self._omega_0 * np.sqrt(1 - self.damping_ratio**2)
        decay = np.exp(-self.damping_ratio * self._omega_0 * dt)

        phi_1 = 2 * decay * np.cos(omega_d * dt)
        phi_2 = -decay**2

        # The stationary variance of a damped harmonic oscillator driven by white noise
        # with PSD S_0 is: var_target = S_0 / (4*zeta*omega_0^3)
        #
        # The AR(2) process x_t = φ_1*x_{t-1} + φ_2*x_{t-2} + ε_t with unit noise variance
        # has stationary variance: var_ar = (1-φ_2) / ((1+φ_2)*(1-φ_2-φ_1)*(1-φ_2+φ_1))
        #
        # To achieve the target variance, we need noise_std = sqrt(var_target / var_ar)
        var_target = self.driving_psd / (4 * self.damping_ratio * self.omega_0**3)
        var_ar = (1 - phi_2) / ((1 + phi_2) * (1 - phi_2 - phi_1) * (1 - phi_2 + phi_1))

        noise_std = np.sqrt(var_target / var_ar)

        return np.array([phi_1, phi_2]), noise_std

    def evolve_until(self, t):
        '''Evolve the oscillator until time t.

        This method computes AR(2) coefficients on the fly for the exact
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

        # Compute AR coefficients for this specific timestep
        ar_coefs, noise_std = self._compute_ar_coefficients(delta_t)

        # Update state using the computed coefficients
        new_val = np.dot(ar_coefs, self._state)
        new_val += noise_std * self.rng.standard_normal()

        # Update state (shift and insert new value)
        self._state[1] = self._state[0]
        self._state[0] = new_val

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
