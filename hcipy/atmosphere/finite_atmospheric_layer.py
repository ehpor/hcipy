from .atmospheric_model import AtmosphericLayer, power_spectral_density_von_karman, fried_parameter_from_Cn_squared
import numpy as np
from ..statistics import SpectralNoiseFactoryMultiscale

class FiniteAtmosphericLayer(AtmosphericLayer):
	def __init__(self, input_grid, Cn_squared=None, L0=np.inf, velocity=0, height=0, oversampling=2):
		self._dirty = True
		self._achromatic_screen = None

		AtmosphericLayer.__init__(self, input_grid, Cn_squared, L0, velocity, height)

		self.oversampling = oversampling
		self.center = np.zeros(2)

		self.reset()
	
	def reset(self):
		self.psd = power_spectral_density_von_karman(fried_parameter_from_Cn_squared(self.Cn_squared, 1), self.L0)

		self.noise_factory = SpectralNoiseFactoryMultiscale(self.psd, self.input_grid, self.oversampling)
		self.noise = self.noise_factory.make_random()

		self._dirty = False
	
	def phase_for(self, wavelength):
		if self._dirty:
			self.reset()
		
		if self._achromatic_screen is None:
			self._achromatic_screen = self.noise.shifted(self.center)()
		
		return self._achromatic_screen / wavelength
	
	def evolve_until(self, t):
		self.center = self.velocity * t
		self._achromatic_screen = None
	
	@property
	def Cn_squared(self):
		return self._Cn_squared
	
	@Cn_squared.setter
	def Cn_squared(self, Cn_squared):
		self._Cn_squared = Cn_squared
		self._dirty = True
	
	@property
	def outer_scale(self):
		return self._L0
	
	@outer_scale.setter
	def L0(self, L0):
		self._L0 = L0
		self._dirty = True