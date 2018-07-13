import numpy as np
import matplotlib.pyplot as plt

from ..field import make_pupil_grid
from ..optics import DeformableMirror
from ..mode_basis import ModeBasis
from ..math_util import inverse_truncated
from .calibration import calibrate_modal_reconstructor

class PIDController(object):
	def __init__(self, gain_p,gain_i,gain_d,interaction_matrix=None, reference=0):
		self.flr = None
		self.error=0
		self.previous_error=0
		self.integrator=0
		self.derivative=0
		self.gain_p=gain_p
		self.gain_i=gain_i
		self.gain_d=gain_d
		self.reference = 0
		self.actuators = 0

		self.interaction_matrix = interaction_matrix
	
	def submit_wavefront(self, t, filtered_wf, filtered_cov, wfs_number=0):
		self.error = filtered_wf - self.reference
		self.integrator-=self.error
		self.derivative=self.error-self.previous_error

		self.actuators =  self.gain_p * self.interaction_matrix.dot(self.error) +self.gain_i * self.interaction_matrix.dot(self.integrator) +self.gain_d * self.interaction_matrix.dot(self.derivative)
	
		self.previous_error=self.error
	def get_actuators(self, t, dm_number=0):
		return self.actuators
	
	@property
	def interaction_matrix(self):
		return self._interaction_matrix
	
	@interaction_matrix.setter
	def interaction_matrix(self, interaction_matrix):
		self._interaction_matrix = interaction_matrix
