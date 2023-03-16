from .atmospheric_model import AtmosphericLayer, power_spectral_density_von_karman, fried_parameter_from_Cn_squared
import numpy as np
from ..util import SpectralNoiseFactoryMultiscale

import copy

class FiniteAtmosphericLayer(AtmosphericLayer):
	def __init__(self, input_grid, Cn_squared=None, L0=np.inf, velocity=0, height=0, oversampling=2, seed=None):
		self._noise = None
		self._achromatic_screen = None

		AtmosphericLayer.__init__(self, input_grid, Cn_squared, L0, velocity, height)

		self.oversampling = oversampling
		self.center = np.zeros(2)

		self._original_rng = np.random.default_rng(seed)

		self.reset()

	def reset(self, make_independent_realization=False):
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
		self._noise = self.noise_factory.make_random()

		self._achromatic_screen = None

	@property
	def noise(self):
		if self._noise is None:
			self.reset()

		return self._noise

	@property
	def achromatic_screen(self):
		if self._achromatic_screen is None:
			self._achromatic_screen = self.noise.shifted(self.center)()

		return self._achromatic_screen

	def phase_for(self, wavelength):
		return self.achromatic_screen / wavelength

	def evolve_until(self, t):
		self.center = self.velocity * t
		self._achromatic_screen = None

	@property
	def Cn_squared(self):  # noqa: N802
		return self._Cn_squared

	@Cn_squared.setter
	def Cn_squared(self, Cn_squared):  # noqa: N802
		self._Cn_squared = Cn_squared
		self._noise = None

	@property
	def outer_scale(self):
		return self._L0

	@outer_scale.setter
	def L0(self, L0):  # noqa: N802
		self._L0 = L0
		self._noise = None
