import numpy as np
from .detector import Detector
from .optical_element import OpticalElement
from .wavefront import Wavefront
from ..mode_basis import ModeBasis, make_gaussian_laguerre_basis
from ..field import Field

def blockshaped(arr, nrows, ncols):
	"""
	Return an array of shape (n, nrows, ncols) where
	n * nrows * ncols = arr.size

	If arr is a 2D array, the returned array looks like n subblocks with
	each subblock preserving the "physical" layout of arr.
	"""
	h, w = arr.shape
	return (arr.reshape(h//nrows, nrows, -1, ncols)
				.swapaxes(1,2)
				.reshape(-1, nrows, ncols))


def unblockshaped(arr, h, w):
	"""
	Return an array of shape (h, w) where
	h * w = arr.size

	If arr is of shape (n, nrows, ncols), n sublocks of shape (nrows, ncols),
	then the returned array preserves the "physical" layout of the sublocks.
	"""
	n, nrows, ncols = arr.shape
	return (arr.reshape(h//nrows, -1, nrows, ncols)
				.swapaxes(1,2)
				.reshape(h, w))

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

class MultiModeFiber(Detector):
	def __init__(self, input_grid, num_modes, mode_field_diameter, modes=None):
		self.input_grid = input_grid
		self.mode_field_diameter = mode_field_diameter
		self.num_modes = num_modes

		if modes is None:
			modes = make_gaussian_laguerre_basis
		
		self.modes = modes( input_grid, self.num_modes, mode_field_diameter)
		self.intensity = 0

	def integrate(self, wavefront, dt, weight=1):
		# Basis decomposition
		basis_coef = np.dot(self.modes.transformation_matrix.T, wavefront.electric_field * wavefront.electric_field.grid.weights)

		# Scrambling matrix
		# TODO : implement random unitary matrix generation
		scrambled_output = basis_coef

		# Reconstruct the electric field at the fiber output
		electric_field_out =  np.dot(self.modes.transformation_matrix, scrambled_output)

		# Measure the electric field output
		self.intensity += weight * dt * abs(electric_field_out)**2
	
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

# This implementation assumes orthogonality of the different fibers.
# Forward() should be changed if this is added in the future.
# Also, the modes are independent of wavelength.
class MultiModeFiberArray(OpticalElement):
	def __init__(self, input_grid, fiber_grid, num_modes, *args, modes = None, **kwargs):
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

		if modes is None:
			modes = make_gaussian_laguerre_basis
		self.num_modes = num_modes
		print("test")
		#num_modes, mode_field_diameter, grid
		self.fiber_modes = [modes(input_grid.shifted(-p), self.num_modes, *args, **kwargs) for p in fiber_grid]
		self.fiber_modes = ModeBasis(np.vstack(self.fiber_modes))
		print(self.fiber_modes.transformation_matrix.shape)

		self.projection_matrix = self.fiber_modes.transformation_matrix
		self.projection_matrix = self.projection_matrix.reshape((input_grid.size,-1))
		print(self.projection_matrix.shape)

	def forward(self, wavefront):
		print(self.projection_matrix.shape)
		print(wavefront.electric_field.shape)

		res = self.projection_matrix.T.dot(wavefront.electric_field * self.input_grid.weights).reshape((self.fiber_grid.size,-1))
		return Wavefront(Field(res, self.fiber_grid), wavefront.wavelength)
	
	def backward(self, wavefront):
		res = self.projection_matrix.dot(wavefront.electric_field).reshape((self.fiber_grid.size,-1))
		return Wavefront(Field(res, self.input_grid), wavefront.wavelength)
	
	def get_transformation_matrix_forward(self, wavelength=1):
		return self.projection_matrix.T
	
	def get_transformation_matrix_backward(self, wavelength=1):
		return self.projection_matrix

