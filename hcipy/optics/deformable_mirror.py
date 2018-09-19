import numpy as np
from .optical_element import OpticalElement
from ..field import Field
from ..mode_basis import ModeBasis

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
		return 2 * self.surface * wavefront.wavenumber