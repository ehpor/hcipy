import numpy as np
from .optical_element import OpticalElement
from ..field import Field
from ..mode_basis import ModeBasis

class DeformableMirror(OpticalElement):
	def __init__(self, influence_functions):
		self.influence_functions = influence_functions
		self.actuators = np.zeros(len(influence_functions))
		self.input_grid = influence_functions[0].grid
	
	def forward(self, wavefront):
		wf = wavefront.copy()
		wf.electric_field *= np.exp(2j * self.surface * wavefront.wavenumber)
		return wf
	
	def backward(self, wavefront):
		wf = wavefront.copy()
		wf.electric_field *= np.exp(-2j * self.surface * wavefront.wavenumber)
		return wf
	
	@property
	def influence_functions(self):
		return self._influence_functions
	
	@influence_functions.setter
	def influence_functions(self, influence_functions):
		self._influence_functions = ModeBasis(influence_functions)
		self._transformation_matrix = self._influence_functions.transformation_matrix
	
	@property
	def surface(self):
		surf = self._transformation_matrix.dot(self.actuators)
		return Field(surf, self.input_grid)