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
		modes = ModeBasis([Field(input_grid.x, input_grid), Field(input_grid.y, input_grid)], input_grid)

		super().__init__(modes)
