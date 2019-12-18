import numpy as np
from .propagator import Propagator
from ..optics import Wavefront, AgnosticOpticalElement, make_agnostic_forward, make_agnostic_backward
from ..field import Field
from ..fourier import FastFourierTransform
from ..field import evaluate_supersampled, make_pupil_grid, subsample_field

class FresnelPropagator(AgnosticOpticalElement):
	'''The monochromatic Fresnel propagator for scalar fields.

	The Fresnel propagator is implemented as described in [1]_.

	.. [1] Goodman, J.W., 2005 Introduction to Fourier optics. Roberts and Company Publishers.

	Parameters
	----------
	input_grid : anything
		This argument is ignored. The input grid is taken from the incoming wavefront.
	distance : scalar
		The distance to propagate
	num_oversampling : int
		The number of times the transfer function is oversampled. Default is 2.
	wavelength : scalar
		The wavelength of the wavefront.
	refractive_index : scalar
		The refractive index of the medium that the wavefront is propagating in.

	Raises
	------
	ValueError
		If the `input_grid` is not regular and Cartesian.
	'''
	def __init__(self, input_grid, distance, num_oversampling=2, refractive_index=1):
		self._distance = distance

		self._num_oversampling = num_oversampling
		self._refractive_index = refractive_index

		AgnosticOpticalElement.__init__(self, grid_dependent=True, wavelength_dependent=True)

	def make_instance(self, instance_data, input_grid, output_grid, wavelength):
		if not input_grid.is_regular or not input_grid.is_('cartesian'):
			raise ValueError('The input grid must be a regular, Cartesian grid.')
	
		instance_data.fft = FastFourierTransform(input_grid, q=2)
		
		k = 2 * np.pi / wavelength * self.evaluate_parameter(self.refractive_index, input_grid, output_grid, wavelength)
		L_max = np.max(input_grid.dims * input_grid.delta)

		if np.any(input_grid.delta < wavelength * self.distance / L_max):
			enlarged_input_grid = make_pupil_grid(2 * input_grid.dims, 2 * input_grid.delta * (input_grid.dims - 1))
			fft_up_scale = FastFourierTransform(enlarged_input_grid)

			def impulse_response_generator(grid):
				r_squared = grid.x**2 + grid.y**2
				return Field(np.exp(1j * k * self.distance) / (1j * wavelength * self.distance) * np.exp(1j * k * r_squared / (2 * self.distance)), grid)

			impulse_response = evaluate_supersampled(impulse_response_generator, enlarged_input_grid, self.num_oversampling)

			instance_data.transfer_function = fft_up_scale.forward(impulse_response)
		else:
			def transfer_function_generator(grid):
				k_squared = grid.as_('polar').r**2
				phase_factor = np.exp(1j * k * self.distance)
				return Field(np.exp(-0.5j * self.distance * k_squared / k) * phase_factor, grid)

			instance_data.transfer_function = evaluate_supersampled(transfer_function_generator, instance_data.fft.output_grid, self.num_oversampling)

	@property
	def distance(self):
		return self._distance

	@distance.setter
	def distance(self, distance):
		self._distance = distance

		self.clear_cache()

	@property
	def num_oversampling(self):
		return self._num_oversampling

	@num_oversampling.setter
	def num_oversampling(self, num_oversampling):
		self._num_oversampling = num_oversampling

		self.clear_cache()

	@property
	def refractive_index(self):
		return self._refractive_index

	@refractive_index.setter
	def refractive_index(self, refractive_index):
		self._refractive_index = refractive_index

		self.clear_cache()

	def get_input_grid(self, output_grid, wavelength):
		return output_grid
	
	def get_output_grid(self, input_grid, wavelength):
		return input_grid

	@make_agnostic_forward
	def forward(self, instance_data, wavefront):
		'''Propagate a wavefront forward by a certain distance.
	
		Parameters
		----------
		wavefront : Wavefront
			The incoming wavefront.
		
		Returns
		-------
		Wavefront
			The wavefront after the propagation.
		'''
		ft = instance_data.fft.forward(wavefront.electric_field)
		ft *= instance_data.transfer_function

		return Wavefront(instance_data.fft.backward(ft), wavefront.wavelength, wavefront.input_stokes_vector)

	@make_agnostic_backward
	def backward(self, instance_data, wavefront):
		'''Propagate a wavefront backward by a certain distance.
	
		Parameters
		----------
		wavefront : Wavefront
			The incoming wavefront.
		
		Returns
		-------
		Wavefront
			The wavefront after the propagation.
		'''
		ft = instance_data.fft.forward(wavefront.electric_field)
		ft *= np.conj(instance_data.transfer_function)

		return Wavefront(instance_data.fft.backward(ft), wavefront.wavelength, wavefront.input_stokes_vector)
