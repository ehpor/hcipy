from .atmospheric_model import AtmosphericLayer
from ..util import inverse_tikhonov

class ModalAdaptiveOpticsLayer(AtmosphericLayer):
    """An atmospheric layer that simulates modal adaptive optics correction.

    This layer wraps an existing atmospheric layer and applies a correction
    based on a set of controlled modes, simulating the effect of an adaptive
    optics system with a certain framerate and lag.

    Note
    ----
    This class provides a rudimentary way to simulate AO-system-like residuals
    in an infinite signal-to-noise ratio (SNR) regime. It is not intended as a
    first-principles simulation of an AO system, but rather as a perfect
    representation of an AO system's ability to correct modes.

    Parameters
    ----------
    layer : AtmosphericLayer
        The underlying atmospheric layer to be corrected.
    controlled_modes : ModeBasis
        The mode basis used for adaptive optics correction.
    lag : int
        The number of frames of lag in the adaptive optics system. This has
        to be an integer; if it's not, it will be rounded to the nearest integer.
    framerate : scalar or None
        The framerate of the adaptive optics system in 1/time. If this is None,
        the layer will be reconstructed every call to `evolve_until()`.
    """
    def __init__(self, layer, controlled_modes, lag, framerate=None):
        self.layer = layer

        super().__init__(layer.input_grid, layer.Cn_squared, layer.L0, layer.velocity, layer.height)

        self.transformation_matrix = controlled_modes.transformation_matrix
        self.transformation_matrix_inverse = inverse_tikhonov(self.transformation_matrix, 1e-7)

        self.controlled_modes = controlled_modes
        self.corrected_coeffs = []
        self.lag = round(lag)
        self.framerate = framerate

        # Initialize reconstruction.
        self._reconstruct_wavefront()

    @property
    def framerate(self):
        return self._framerate

    @framerate.setter
    def framerate(self, framerate):
        self._framerate = framerate

        if self.framerate is not None:
            self._last_reconstruction_frame = int(self.layer.t * framerate)

    def phase_for(self, wavelength):
        """Calculates the phase screen for a given wavelength with AO correction.

        Parameters
        ----------
        wavelength : scalar
            The wavelength for which to calculate the phase screen.

        Returns
        -------
        Field
            The phase screen with adaptive optics correction applied.
        """
        ps = self.layer.phase_for(wavelength)
        ps -= self.transformation_matrix.dot(self.corrected_coeffs[0] / wavelength)

        return ps

    def evolve_until(self, t):
        """Evolves the atmospheric layer until a certain time `t`.

        This method evolves the underlying atmospheric layer in discrete steps
        determined by the `framerate`, and calls `reconstruct_wavefront` at
        each of these steps.

        Parameters
        ----------
        t : scalar
            The time until which to evolve the layer.
        """
        if self.framerate is None:
            self.layer.evolve_until(t)
            self._reconstruct_wavefront()

            return

        dt = 1 / self.framerate

        # Evolve and reconstruct in steps, performing reconstruction along the way.
        while self.layer.t < t:
            next_reconstruction_t = (self._last_reconstruction_frame + 1) * dt
            next_t = min(next_reconstruction_t, t)

            self.layer.evolve_until(next_t)

            if next_reconstruction_t <= t:
                self._reconstruct_wavefront()

        self._t = t

    def _reconstruct_wavefront(self):
        """Reconstructs the wavefront and updates the corrected coefficients.

        This method simulates the wavefront sensing and control part of an
        adaptive optics system. It reconstructs the wavefront from the current
        atmospheric layer and updates the internal buffer of corrected coefficients
        considering the system's lag.
        """
        coeffs = self.transformation_matrix_inverse.dot(self.layer.phase_for(1))
        if len(self.corrected_coeffs) > self.lag:
            self.corrected_coeffs.pop(0)
        self.corrected_coeffs.append(coeffs)

        if self.framerate is not None:
            self._last_reconstruction_frame = int(self.layer.t * self.framerate)

    @property
    def Cn_squared(self):  # noqa: N802
        """The atmospheric refractive index structure constant (Cn^2) of the layer.

        Returns
        -------
        scalar
            The Cn^2 value.
        """
        return self.layer.Cn_squared

    @Cn_squared.setter
    def Cn_squared(self, Cn_squared):  # noqa: N802
        self.layer.Cn_squared = Cn_squared

    @property
    def outer_scale(self):
        """The outer scale of turbulence (L0) of the layer.

        Returns
        -------
        scalar
            The L0 value.
        """
        return self.layer.L0

    @outer_scale.setter
    def L0(self, L0):  # noqa: N802
        self.layer.L0 = L0

    def reset(self):
        """Resets the corrected coefficients and the underlying atmospheric layer.
        """
        self.corrected_coeffs = []
        self.layer.reset()

        self._reconstruct_wavefront()
