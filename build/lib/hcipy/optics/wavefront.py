import copy
import numpy as np

from ..field import Field

# TODO Should add a pilot Gaussian beam with each Wavefront

class Wavefront(object):
	def __init__(self, electric_field, wavelength=1):
		self.electric_field = electric_field
		self.wavelength = wavelength
	
	def copy(self):
		return copy.deepcopy(self)
	
	@property
	def electric_field(self):
		return self._electric_field
	
	@electric_field.setter
	def electric_field(self, U):
		if hasattr(U, 'grid'):
			self._electric_field = U.astype('complex')
		else:
			if len(U) == 2:
				self._electric_field = Field(U[0].astype('complex'), U[1])
			else:
				raise ValueError("Electric field requires an accompanying grid.")
	
	@property
	def wavenumber(self):
		return 2*np.pi / self.wavelength
	
	@wavenumber.setter
	def wavenumber(self, wavenumber):
		self.wavelength = 2*np.pi / wavenumber
	
	@property
	def grid(self):
		return self.electric_field.grid
	
	@property
	def intensity(self):
		return np.abs(self.electric_field)**2

	@property
	def amplitude(self):
		return np.abs(self.electric_field)

	@property
	def phase(self):
		phase = np.angle(self.electric_field)
		return Field(phase, self.electric_field.grid)
	
	@property
	def real(self):
		return np.real(self.electric_field)
	
	@property
	def imag(self):
		return np.imag(self.electric_field)
	
	@property
	def power(self):
		return self.intensity * self.grid.weights
	
	@property
	def total_power(self):
		return np.sum(self.power)
	
	@total_power.setter
	def total_power(self, p):
		self.electric_field *= np.sqrt(p / self.total_power)