import numpy as np
import matplotlib.pyplot as plt

from ..field import make_pupil_grid
from ..optics import DeformableMirror
from ..mode_basis import ModeBasis
from ..math_util import inverse_truncated
from .modal_reconstructor import calibrate_modal_reconstructor

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

def make_interaction_matrix(controller, dm, wavefront, wavefront_sensor, wavefront_estimator, amplitude=0.005, f=None):
	wf = wavefront.copy()

	if f is None:
		interaction_matrix = calibrate_modal_reconstructor(f, num_modes, wf, wavefront_sensor, wavefront_estimator, amplitude)
	else:
		num_modes = len(dm.actuators)
		influence_functions = []
		
		wf.total_power = 1

		# Let's get the proper lenslet measurements we want. This should really be done by the wavefront sensor estimator
		img = wavefront_sensor(wf).intensity
		ref = wavefront_estimator.estimate([img]).ravel()

		influence_functions = []
		for i in range(num_modes):
			total_slopes = np.zeros(ref.shape)

			for amp in np.array([-1*amplitude, amplitude]):
		   
				act_levels = np.zeros(num_modes)
				act_levels[i] = wf.wavelength * amp
	
				dm.actuators = act_levels
				wfs_img = wavefront_sensor(dm(wf)).intensity
		 
				slopes = wavefront_estimator.estimate([wfs_img]).ravel()
				total_slopes += (slopes - ref) / (2 * amp)
			influence_functions.append(f.estimate(total_slopes, 0))
		
		influence_functions = ModeBasis(influence_functions)
		interaction_matrix = inverse_truncated(influence_functions.transformation_matrix)

	controller.interaction_matrix = interaction_matrix