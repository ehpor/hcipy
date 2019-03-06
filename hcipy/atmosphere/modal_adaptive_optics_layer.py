from .atmospheric_model import AtmosphericLayer
from ..math_util import inverse_tikhonov

import numpy as np

class ModalAdaptiveOpticsLayer(AtmosphericLayer):
	def __init__(self, layer, controlled_modes, lag):
		self.layer = layer
		self.controlled_modes = controlled_modes

		AtmosphericLayer.__init__(self, layer.input_grid, layer.Cn_squared, layer.L0, layer.velocity, layer.height)

		self.transformation_matrix = controlled_modes.transformation_matrix
		self.transformation_matrix_inverse = inverse_tikhonov(self.transformation_matrix, 1e-7)

		self.corrected_coeffs = []
		self.lag = lag

	def phase_for(self, wavelength):
		ps = self.layer.phase_for(wavelength)
		ps -= self.transformation_matrix.dot(self.corrected_coeffs[0] / wavelength)

		return ps
	
	def evolve_until(self, t):
		self.layer.evolve_until(t)

		coeffs = self.transformation_matrix_inverse.dot(self.layer.phase_for(1))
		if len(self.corrected_coeffs) > self.lag:
			self.corrected_coeffs.pop(0)
		self.corrected_coeffs.append(coeffs)
	
	@property
	def Cn_squared(self):
		return self.layer.Cn_squared
	
	@Cn_squared.setter
	def Cn_squared(self, Cn_squared):
		self.layer.Cn_squared = Cn_squared
	
	@property
	def outer_scale(self):
		return self.layer.L0

	@outer_scale.setter
	def L0(self, L0):
		self.layer.L0 = L0
	
	def reset(self):
		self.corrected_coeffs = []
		self.layer.reset()