import numpy as np
from ..propagation import FraunhoferPropagator
from ..math_util import inverse_tikhonov, SVD

class ElectricFieldConjugation(object):
	def __init__(self, wavefront, dark_hole_grid, influence_functions=None, propagator=None, matrix_invertor=None):
		wf = wavefront.copy()

		if propagator is None:
			propagator = FraunhoferPropagator(wavefront.electric_field.grid, dark_hole_grid)

		matrix = propagator.get_transformation_matrix_forward(wavefront.wavelength)
		matrix *= 1j * wavefront.electric_field * 2*np.pi
		matrix = np.vstack((matrix.real, matrix.imag))

		if influence_functions is None:
			self.linearized_propagator = matrix
		else:
			self.linearized_propagator = matrix.dot(influence_functions.transformation_matrix)
		
		if matrix_invertor is None:
			matrix_invertor = lambda lin_prop, svd: inverse_tikhonov(lin_prop, 1e-2, svd=svd)
		self.matrix_invertor = matrix_invertor

		self._svd = None
	
	@property
	def efc_matrix(self):
		if self._efc_matrix is None:
			self._efc_matrix = -self.matrix_invertor(self.linearized_propagator, self.svd)
		
		return self._efc_matrix
	
	@property
	def matrix_invertor(self):
		return self._matrix_invertor
	
	@matrix_invertor.setter
	def matrix_invertor(self, matrix_invertor):
		self._efc_matrix = None
		self._matrix_invertor = matrix_invertor
	
	@property
	def svd(self):
		if self._svd is None:
			self._svd = SVD(self.linearized_propagator)
		return self._svd

	def get_mode_basis(self, num_modes=-1):
		return 

	def single_step(self):
		pass
