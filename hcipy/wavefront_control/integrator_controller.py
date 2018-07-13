import numpy as np
import matplotlib.pyplot as plt

from ..field import make_pupil_grid
from ..optics import DeformableMirror
from ..mode_basis import ModeBasis
from ..math_util import inverse_truncated
from .calibration import calibrate_modal_reconstructor

class IntegratorController(object):
	def __init__(self, gain, interaction_matrix=None, leakage=0, reference=0):
		self.flr = None
		self.gain = gain
		self.leakage = leakage

		self.reference = 0
		self.actuators = 0

		self.interaction_matrix = interaction_matrix
	
	def submit_wavefront(self, t, filtered_wf, filtered_cov, wfs_number=0):
		self.error = filtered_wf - self.reference
		self.actuators = (1 - self.leakage) * self.actuators - self.gain * self.interaction_matrix.dot(self.error)
	
	def get_actuators(self, t, dm_number=0):
		return self.actuators
	
	@property
	def interaction_matrix(self):
		return self._interaction_matrix
	
	@interaction_matrix.setter
	def interaction_matrix(self, interaction_matrix):
		self._interaction_matrix = interaction_matrix
