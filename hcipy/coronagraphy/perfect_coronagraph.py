import numpy as np
from ..optics import OpticalElement
from ..mode_basis import ModeBasis
from ..math_util import inverse_truncated

class PerfectCoronagraph(OpticalElement):
	def __init__(self, aperture, order=2, coeffs=None):
		self.pupil_grid = aperture.grid
		modes = []

		if coeffs is not None:
			order = int(2 * np.ceil(0.5 * (np.sqrt(8*len(coeffs) + 1) - 1)))
			self.coeffs = coeffs
		else:
			self.coeffs = np.ones(int(order * (order / 2 + 1) / 4))

		for i in range(order // 2):
			for j in range(i + 1):
				modes.append(aperture * self.pupil_grid.x**j * self.pupil_grid.y**(i-j))
		
		self.mode_basis = ModeBasis(modes).orthogonalized

		self.transformation = self.mode_basis.transformation_matrix
		self.transformation_inverse = inverse_truncated(self.transformation, 1e-6)

	def forward(self, wavefront):
		wf = wavefront.copy()

		correction = self.transformation.dot(self.coeffs * self.transformation_inverse.dot(wf.electric_field))
		wf.electric_field -= correction

		return wf
	
	def backward(self, wavefront):
		return self.forward(wavefront)
	
	def get_transformation_matrix_forward(self, wavelength=1):
		return np.eye(self.pupil_grid.size) - self.transformation.dot(self.coeffs * self.transformation_inverse)
	
	def get_transformation_matrix_backward(self, wavelength=1):
		return self.get_transformation_matrix_forward(wavelength)