import numpy as np
from .detector import Detector
from .optical_element import OpticalElement, AgnosticOpticalElement, make_agnostic_forward, make_agnostic_backward
from .wavefront import Wavefront
from ..mode_basis import ModeBasis, make_LP_modes
from ..field import Field

class StepIndexFiber(AgnosticOpticalElement):
	'''A step-index fiber.

	The modal behaviour of the step-index fiber depends on the input parameters.

	Parameters
	----------
	core_radius : scalar
		The radius of the high-index core.
	NA : scalar or function of wavelength
		The numerical aperture of the fiber.
	fiber_length : scalar
		The length of the optical fiber.
	'''
	def __init__(self, core_radius, NA, fiber_length):
		self._core_radius = core_radius
		self._NA = NA
		self.fiber_length = fiber_length
		
		AgnosticOpticalElement.__init__(self, False, True)
	
	def make_instance(self, instance_data, input_grid, output_grid, wavelength):
		monochromatic_V = self.V(wavelength)
		instance_data.NA = self.evaluate_parameter(self._NA, input_grid, output_grid, wavelength)
		instance_data.fiber_modes, instance_data.beta = make_LP_modes(input_grid, monochromatic_V, self.core_radius, return_betas=True)

	def num_modes(self, wavelength):
		'''The approximate amount of modes of the fiber.
		'''
		V = self.V(wavelength)
		return V**2 / 2

	def V(self, wavelength):
		'''The normalized frequency parameter for step-index fibers.
		'''
		return 2 * np.pi / wavelength * self.core_radius * self.NA

	def mode_field_radius(self, wavelength):
		'''The mode field radius of the fiber.
		
		The mode field radius is the radius of the gaussian beam best matched to the fundamental mode [Marcuse1977]_.
		
		.. [Marcuse1977] D. Marcuse 1977, "Loss analysis of single-mode fiber splices," The Bell System Technical Journal 56, 703-718 (2014) 
		'''
		V = self.V(wavelength)
		w = self.core_radius * (0.65 + 1.619 / V**(3 / 2) + 2.879 / V**6)
		return w

	@property
	def core_radius(self):
		'''The core radius of this fiber.
		'''
		return self._core_radius
	
	@core_radius.setter
	def core_radius(self, core_radius):
		self._core_radius = core_radius
		self.clear_cache()
	
	@property
	def NA(self):
		'''The numerical aperture of this fiber.
		'''
		return self._NA

	@NA.setter
	def NA(self, NA):
		self._NA = NA
		self.clear_cache()

	def get_input_grid(self, output_grid, wavelength):
		return output_grid

	def get_output_grid(self, input_grid, wavelength):
		return input_grid

	@make_agnostic_forward
	def modal_decomposition(self, instance_data, wavefront):
		'''Decompose the input wavefront into the modal distribution of the fiber.

		Parameters
		----------
		wavefront : Wavefront
			The incoming wavefront.
		
		Returns
		-------
		array_like
			The modal coefficients.
		'''
		
		M = instance_data.fiber_modes.transformation_matrix
		mode_coefficients = np.einsum('...i, i, ij->...j', wavefront.electric_field, wavefront.grid.weights, M.conj())
		return mode_coefficients

	@make_agnostic_forward
	def forward(self, instance_data, wavefront):
		'''Forward propagate the light through the fiber.

		Parameters
		----------
		wavefront : Wavefront
			The incoming wavefront.
		
		Returns
		-------
		Wavefront
			The wavefront that exits the fiber.
		'''
				
		M = instance_data.fiber_modes.transformation_matrix
		mode_coefficients = np.einsum('...i, i, ij->...j', wavefront.electric_field, wavefront.grid.weights, M.conj())
		output_electric_field = Field(np.einsum('...i, i, ij->...j', mode_coefficients, np.exp(1j * instance_data.beta * self.fiber_length), M.T.conj()), wavefront.grid)
		
		return Wavefront(output_electric_field, wavefront.wavelength)

	@make_agnostic_backward
	def backward(self, instance_data, wavefront):
		'''Backward propagate the light through the fiber.

		Parameters
		----------
		wavefront : Wavefront
			The incoming wavefront.
		
		Returns
		-------
		Wavefront
			The wavefront that exits the fiber.
		'''
		M = instance_data.fiber_modes.transformation_matrix
		mode_coefficients = np.einsum('...i, i, ij->...j', wavefront.electric_field, wavefront.grid.weights, M.conj())
		output_electric_field = Field(np.einsum('...i, i, ij->...j', mode_coefficients, np.exp(-1j * instance_data.beta * self.fiber_length), M.T.conj()), wavefront.grid)
		
		return Wavefront(output_electric_field, wavefront.wavelength)

class SingleModeFiber(Detector):
	def __init__(self, input_grid, mode_field_diameter, mode=None):
		self.input_grid = input_grid
		self.mode_field_diameter = mode_field_diameter

		if mode is None:
			mode = gaussian_mode
		
		self.mode = mode(self.input_grid, mode_field_diameter)
		self.mode /= np.sum(np.abs(self.mode)**2 * self.input_grid.weights)
		self.intensity = 0

	def integrate(self, wavefront, dt, weight=1):
		self.intensity += weight * dt * (np.dot(wavefront.electric_field * wavefront.electric_field.grid.weights, self.mode))**2
	
	def read_out(self):
		intensity = self.intensity
		self.intensity = 0
		return intensity

# This implementation assumes orthogonality of the different fibers.
# Forward() should be changed if this is added in the future.
# Also, the modes are independent of wavelength.
class SingleModeFiberArray(OpticalElement):
	def __init__(self, input_grid, fiber_grid, mode, *args, **kwargs):
		'''An array of single-mode fibers.

		Parameters
		----------
		input_grid : Grid
			The grid on which the input wavefront is defined.
		fiber_grid : Grid
			The centers of each of the single-mode fibers.
		mode : function
			The mode of the single-mode fibers. The function should take a grid 
			and return the amplitude of the fiber mode.
		'''
		self.input_grid = input_grid
		self.fiber_grid = fiber_grid

		self.fiber_modes = [mode(input_grid.shifted(-p), *args, **kwargs) for p in fiber_grid]
		self.fiber_modes = [mode / np.sqrt(np.sum(np.abs(mode)**2 * input_grid.weights)) for mode in self.fiber_modes]
		self.fiber_modes = ModeBasis(self.fiber_modes)

		self.projection_matrix = self.fiber_modes.transformation_matrix

	def forward(self, wavefront):
		res = self.projection_matrix.T.dot(wavefront.electric_field * self.input_grid.weights)
		return Wavefront(Field(res, self.fiber_grid), wavefront.wavelength)
	
	def backward(self, wavefront):
		res = self.projection_matrix.dot(wavefront.electric_field)
		return Wavefront(Field(res, self.input_grid), wavefront.wavelength)
	
	def get_transformation_matrix_forward(self, wavelength=1):
		return self.projection_matrix.T
	
	def get_transformation_matrix_backward(self, wavelength=1):
		return self.projection_matrix
