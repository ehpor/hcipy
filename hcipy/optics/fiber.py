import numpy as np
from .detector import Detector
from .optical_element import OpticalElement, AgnosticOpticalElement, make_agnostic_forward, make_agnostic_backward
from .wavefront import Wavefront
from ..mode_basis import ModeBasis, make_LP_modes
from ..field import Field

def fiber_mode_gaussian(grid, mode_field_diameter):
	r2 = grid.x**2 + grid.y**2
	return np.exp(-r2/((0.5 * mode_field_diameter)**2))

class StepIndexFiber(AgnosticOpticalElement):
	''' A step-index fiber.

	The modal behaviour of the step-index fiber depends on the input parameters.

	Parameters
	----------
	core_radius : scalar
		The radius of the high-index core.
	NA : scalar
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
		instance_data.fiber_modes, instance_data.beta = make_LP_modes(input_grid, monochromatic_V, self.core_radius, mode_cutoff=None)
		instance_data.Minv = np.linalg.pinv( instance_data.fiber_modes.transformation_matrix )

	def num_modes(self, wavelength):
		V = self.V(wavelength)
		return V**2 / 2

	def V(self, wavelength):
		return 2 * np.pi / wavelength * self.core_radius * self.NA

	def mode_field_radius(self, wavelength):
		V = self.V(wavelength)
		w = self.core_radius * (0.65 + 1.619/V**(3/2) + 2.879/V**6)
		return w

	@property
	def core_radius(self):
		return self._core_radius
	
	@core_radius.setter
	def core_radius(self, core_radius):
		self._core_radius = core_radius
		self.clear_cache()
	
	@property
	def NA(self):
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
		wf = wavefront.copy()
		
		M = instance_data.fiber_modes.transformation_matrix
		#mode_coefficients = instance_data.Minv.dot(wf.electric_field)
		#mode_coefficients = np.sum( M.T.conj().dot(wf.electric_field * wf.grid.weights), axis=-1)
		mode_coefficients = M.dot(wf.electric_field * wf.grid.weights)
		return mode_coefficients

	@make_agnostic_forward
	def forward(self, instance_data, wavefront):
		wf = wavefront.copy()
		
		M = instance_data.fiber_modes.transformation_matrix
		#mode_coefficients = instance_data.Minv.dot(wf.electric_field)#
		#mode_coefficients = np.sum( M.T.conj().dot(wf.electric_field * wf.grid.weights), axis=-1)
		mode_coefficients = M.dot(wf.electric_field.conj() * wf.grid.weights)
		output_electric_field = Field(M.T.dot(mode_coefficients * np.exp(1j*instance_data.beta*self.fiber_length)), wf.grid)

		return Wavefront(output_electric_field, wf.wavelength)

	@make_agnostic_backward
	def backward(self, instance_data, wavefront):
		wf = wavefront.copy()
		
		mode_coefficients = M.dot(wf.electric_field.conj() * wf.grid.weights)
		output_electric_field = Field(M.T.dot(mode_coefficients * np.exp(-1j*instance_data.beta*self.fiber_length)), wf.grid)

		return Wavefront(output_electric_field, wf.wavelength)

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