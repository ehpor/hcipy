import numpy as np
import matplotlib.pyplot as plt
from .controller import Controller

from ..field import make_pupil_grid
from ..optics import DeformableMirror
from ..mode_basis import ModeBasis
from ..math_util import inverse_truncated
from .calibration import calibrate_modal_reconstructor

class IntegratorController(Controller):
	'''A Integrator Controller.

	This class implements an integral controller with the option to have a leaky controller.

	Parameters
	----------
	gain : scalar
		Gain of the integral controller
	command_matrix : ndarray
		The command_matrix 
	leakage : scalar
		Leakage for the integral controller. A small number. 
	reference : ndarray
		The reference the controller is convering to for its steady state.
	
	Attributes
	----------
	actuators : ndarray
		The actuator commands to be given to the actuator (i.e., DM)
	'''
	def __init__(self, gain, command_matrix=None, leakage=0, reference=0):
		self.gain = gain
		self.leakage = leakage

		self.reference = 0
		self.actuators = 0
		self.command_matrix = command_matrix
		self.x=[]
	@property
	def gain(self):
		return self._gain
	
	@gain.setter
	def gain(self, gain):
		self._gain = gain
	@property
	def leakage(self):
		return self._leakage
	
	@leakage.setter
	def leakage(self, leakage):
		self._leakage = leakage

	@property
	def command_matrix(self):
		return self._command_matrix
	
	@command_matrix.setter
	def command_matrix(self, command_matrix):
		self._command_matrix = command_matrix
		
	def submit_wavefront(self, t, wavefront, filtered_cov, wfs_number=0):
		'''Submit a wavefront estimate to integrator.

		Parameters
		----------
		t : scalar
			Time at which the estimate was taken.
		wavefront : Field
			The estimate of the wavefront. This can be slopes, mode coefficients, etc...
		wfs_number : int
			The index of the wavefront sensor. This is meant for support of multiple
			wavefront sensors.
		'''
		self.x=wavefront

	def get_actuators(self, t, dm_number=0):
		'''Get the actuator positions at time `t` for DM number `dm_number`.

		Parameters
		----------
		t : scalar
			The time at which to get the requested actuator positions.
		dm_number : int
			The index of the deformable mirror. This is meant for support for multiple
			deformable mirrors.
		'''
		self.error = self.x - self.reference
		self.actuators = (1 - self.leakage) * self.actuators - self.gain * self.command_matrix.dot(self.error)
		return self.actuators