import numpy as np
from ..mode_basis import ModeBasis
from ..field import Field
from .deformable_mirror import DeformableMirror

class TipTiltMirror(DeformableMirror):
	'''A tip-tilt mirror.

	This deformable mirror class provides a tip-tilt mirror, where the
	actuators are controlled in OPD.

	Parameters
	----------
	input_grid : Grid
		The grid that corresponds to the input wavefront
	'''
	def __init__(self, input_grid):
		self.input_grid = input_grid
		self.actuators = np.zeros((2,))
		
		modes = ModeBasis([Field(self.input_grid.x, self.input_grid), Field(self.input_grid.y, self.input_grid)])
		
		super(TipTiltMirror, self).__init__(modes)