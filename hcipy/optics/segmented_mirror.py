import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse

from .optical_element import OpticalElement
from ..field import Field
from ..plotting import imshow_field
from ..mode_basis import ModeBasis
from .deformable_mirror import DeformableMirror

class SegmentedDeformableMirror(DeformableMirror):
	'''A segmented deformable mirror.

	This deformable mirror class can simulate devices such as those 
	made by IrisAO and BMC. All segments are controlled in piston, 
	tip and tilt.

	Parameters
	----------
	segments : ModeBasis
		A mode basis with all segments.
	'''
	def __init__(self, segments):
		self.segments = segments
		self.actuators = np.zeros(len(segments) * 3)
		self.input_grid = segments.grid
	
	@property
	def segments(self):
		'''The segments of this deformable mirror in a ModeBasis.
		'''
		return self._segments
	
	@segments.setter
	def segments(self, segments):
		self._segments = segments

		tip = []
		tilt = []

		for segment in segments:
			segment_mean = segment.mean()
			norm = np.mean(segment**2) - segment_mean**2

			tip_mode = segment * segment.grid.x
			if norm != 0:
				beta = ((tip_mode * segment).mean() - segment_mean * tip_mode.mean()) / norm
				tip_mode -= beta * segment

			tip_mode = scipy.sparse.csr_matrix(tip_mode)
			tip_mode.eliminate_zeros()
			tip.append(tip_mode)

			tilt_mode = segment * segment.grid.y
			if norm != 0:
				beta = ((tilt_mode * segment).mean() - segment_mean * tilt_mode.mean()) / norm
				tilt_mode -= beta * segment

			tilt_mode = scipy.sparse.csr_matrix(tilt_mode)
			tilt_mode.eliminate_zeros()
			tilt.append(tilt_mode)

		tip = ModeBasis(tip)
		tilt = ModeBasis(tilt)
		
		self.influence_functions = segments + tip + tilt
	
	def get_segment_actuators(self, segment_id):
		'''Get the actuators for an individual segment of the DM.
		
		Parameters
		----------
		segment_id : int
			The index of the segment for which to get the actuators.
		
		Returns
		-------
		piston : scalar
			The piston of the segment in meters.
		tip : scalar
			The tip of the segment in radians.
		tilt : scalar
			The tilt of the segment in radians.
		'''
		piston = self.actuators[segment_id]
		tip = self.actuators[segment_id + len(self._segments)]
		tilt = self.actuators[segment_id + 2 * len(self._segments)]

		return (piston, tip, tilt)

	def set_segment_actuators(self, segment_id, piston, tip, tilt):
		'''Set the actuators for an individual segment of the DM.

		Parameters
		----------
		segment_id : int
			The index of the segment for which to get the actuators.
		piston : scalar
			The piston of the segment in meters.
		tip : scalar
			The tip of the segment in radians.
		tilt : scalar
			The tilt of the segment in radians.
		'''
		self.actuators[segment_id] = piston
		self.actuators[segment_id + len(self._segments)] = tip
		self.actuators[segment_id + 2 * len(self._segments)] = tilt