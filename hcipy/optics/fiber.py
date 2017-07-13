import numpy as np
from .detector import Detector
from .optical_element import OpticalElement
from .wavefront import Wavefront
from ..mode_basis import ModeBasis
from ..field import Field

def fiber_mode_gaussian(grid, mode_field_diameter):
	r2 = grid.x**2 + grid.y**2
	return np.exp(-r2/((0.5 * mode_field_diameter)**2))

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