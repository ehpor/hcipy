from .propagation import MonochromaticPropagator, make_propagator
from ..optics import Wavefront
from ..field import Field

class FraunhoferPropagatorMonochromatic(MonochromaticPropagator):
	def __init__(self, input_grid, output_grid, wavelength_0=1, focal_length=1, wavelength=1):
		from ..fourier import MatrixFourierTransform

		if focal_length is None:
			f_lambda_ref = 1
		else:
			f_lambda_ref = wavelength_0 * focal_length
		
		f_lambda = f_lambda_ref * (wavelength / wavelength_0)

		self.uv_grid = output_grid * ((2*np.pi / f_lambda))
		self.fourier_transform = MatrixFourierTransform(input_grid, uv_grid)
		self.output_grid = output_grid

		self.norm_factor = 1j / f_lambda
		self.input_grid = input_grid

	def forward(self, wavefront):
		U_new = self.fourier_transform(wavefront.electric_field) / self.norm_factor
		return Wavefront(Field(U_new, self.output_grid), wavefront.wavelength)
	
	def backward(self, wavefront):
		U_new = self.fourier_transform.inverse(wavefront.electric_field) * self.norm_factor
		return Wavefront(Field(U_new, self.input_grid), wavefront.wavelength)
	
FraunhoferPropagator = make_propagator(FraunhoferPropagatorMonochromatic)