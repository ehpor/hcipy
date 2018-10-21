import numpy as np
import pkg_resources

from .optical_element import OpticalElement
from ..field import Field, make_uniform_grid
from ..mode_basis import ModeBasis
from ..interpolation import make_linear_interpolator_separated
from ..io import read_fits

def make_xinetics_influence_functions(pupil_grid, num_actuators_across_pupil, actuator_spacing, x_tilt=0, y_tilt=0, z_tilt=0):
	'''Create influence functions for a Xinetics deformable mirror.

	This function uses a The rotation of the deformable mirror will be done in the order X-Y-Z.

	Parameters
	----------
	pupil_grid : Grid
		The grid on which to calculate the influence functions.
	num_actuators_across_pupil : integer
		The number of actuators across the pupil. The total number of actuators will be this number squared.
	actuator_spacing : scalar
		The spacing between actuators before tilting the deformable mirror.
	x_tilt : scalar
		The tilt of the deformable mirror around the x-axis in radians.
	y_tilt : scalar
		The tilt of the deformable mirror around the y-axis in radians.
	z_tilt : scalar
		The tilt of the deformable mirror around the z-axis in radians.

	Returns
	-------
	ModeBasis
		The influence functions for each of the actuators.
	'''
	extent = actuator_spacing * (num_actuators_across_pupil - 1)
	actuator_positions = make_uniform_grid(num_actuators_across_pupil, [extent] * 2)

	evaluated_grid = pupil_grid.scaled(1 / np.cos([y_tilt, x_tilt])).rotated(-z_tilt)

	actuator = np.squeeze(read_fits(pkg_resources.resource_stream('hcipy', 'optics/influence_dm5v2.fits')))
	actuator_grid = make_uniform_grid(actuator.shape, np.array(actuator.shape) * actuator_spacing / 10.0)
	actuator = make_linear_interpolator_separated(actuator.ravel(), actuator_grid, 0)

	modes = [actuator(evaluated_grid.shifted(-p)) for p in actuator_positions]
	modes = [Field(m, pupil_grid) for m in modes]
	return ModeBasis(modes)

class DeformableMirror(OpticalElement):
	'''A deformable mirror using influence functions.

	This class does not contain any temporal simulation (ie. settling time),
	and assumes that there is no crosstalk between actuators.

	Parameters
	----------
	influence_functions : ModeBasis
		The influence function for each of the actuators.
	'''
	def __init__(self, influence_functions):
		self.influence_functions = influence_functions
		self.actuators = np.zeros(len(influence_functions))
		self.input_grid = influence_functions[0].grid
	
	def forward(self, wavefront):
		'''Propagate a wavefront through the deformable mirror.

		Parameters
		----------
		wavefront : Wavefront
			The incoming wavefront.
		
		Returns
		-------
		Wavefront
			The reflected wavefront.
		'''
		wf = wavefront.copy()
		wf.electric_field *= np.exp(2j * self.surface * wavefront.wavenumber)
		return wf
	
	def backward(self, wavefront):
		'''Propagate a wavefront backwards through the deformable mirror.

		Parameters
		----------
		wavefront : Wavefront
			The incoming wavefront.
		
		Returns
		-------
		Wavefront
			The reflected wavefront.
		'''
		wf = wavefront.copy()
		wf.electric_field *= np.exp(-2j * self.surface * wavefront.wavenumber)
		return wf
	
	@property
	def influence_functions(self):
		'''The influence function for each of the actuators of this deformable mirror.
		'''
		return self._influence_functions
	
	@influence_functions.setter
	def influence_functions(self, influence_functions):
		self._influence_functions = ModeBasis(influence_functions)
		self._transformation_matrix = self._influence_functions.transformation_matrix
	
	@property
	def surface(self):
		'''The surface of the deformable mirror in meters.
		'''
		surf = self._transformation_matrix.dot(self.actuators)
		return Field(surf, self.input_grid)
	
	def phase_for(self, wavelength):
		'''Get the phase that is added to a wavefront with a specified wavelength.

		Parameters
		----------
		wavelength : scalar
			The wavelength at which to calculate the phase deformation.
		
		Returns
		-------
		Field
			The calculated phase deformation.
		'''
		return 2 * self.surface * 2*np.pi / wavelength