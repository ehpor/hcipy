import numpy as np
from .propagator import MonochromaticPropagator, make_propagator
from ..optics import Wavefront
from ..field import Field

class FraunhoferPropagatorMonochromatic(MonochromaticPropagator):
	def __init__(self, input_grid, output_grid, wavelength_0=1, focal_length=1, wavelength=1):
		from ..fourier import make_fourier_transform

		if focal_length is None:
			f_lambda_ref = 1
		else:
			f_lambda_ref = wavelength_0 * focal_length
		
		f_lambda = f_lambda_ref * (wavelength / wavelength_0)
		self.uv_grid = output_grid.scaled(2*np.pi / f_lambda)
		self.fourier_transform = make_fourier_transform(input_grid, self.uv_grid)
		self.output_grid = output_grid

		# Intrinsic to Fraunhofer propagation
		self.norm_factor = 1 / (1j * f_lambda)
		self.input_grid = input_grid

	def forward(self, wavefront):
		U_new = self.fourier_transform.forward(wavefront.electric_field) * self.norm_factor
		return Wavefront(Field(U_new, self.output_grid), wavefront.wavelength)
	
	def backward(self, wavefront):
		U_new = self.fourier_transform.backward(wavefront.electric_field) / self.norm_factor
		return Wavefront(Field(U_new, self.input_grid), wavefront.wavelength)
	
	def get_transformation_matrix_forward(self, wavelength=1):
		# Ignore input wavelength and just use the internal one.
		return self.fourier_transform.get_transformation_matrix_forward() * self.norm_factor
	
	def get_transformation_matrix_backward(self, wavelength=1):
		# Ignore input wavelength and just use the internal one.
		return self.fourier_transform.get_transformation_matrix_backward() / self.norm_factor
	
FraunhoferPropagator = make_propagator(FraunhoferPropagatorMonochromatic)