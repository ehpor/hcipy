import numpy as np
import matplotlib.pyplot as plt
from ..field import make_pupil_grid
from ..optics import DeformableMirror
from ..mode_basis import ModeBasis
from ..math_util import inverse_truncated

class Integrator_Controller(object):
	def __init__(self, gain, interaction_matrix=None,leak=None, reference=0):
		
		self.flr=None
		self.gain = gain
		self.leak = leak
		self.reference = 0
		self.actuators = 0
		self.interaction_matrix=interaction_matrix
		if interaction_matrix==None:
			self.interaction_matrix=None
		
			
	def submit_wavefront(self,t, filtered_wf, filtered_cov, wfs_number):
		self.error = (filtered_wf-self.reference) 
		self.actuators=(1-self.leak)*self.actuators- self.gain*self.interaction_matrix.dot(self.error)
	def get_actuators(self, t):
		return self.actuators
	def set_interaction_matrix(self,inter,t):
		self.interaction_matrix=inter

def make_interaction_matrix(controller,dm,wavefront,wavefront_sensor,wavefront_estimator,amplitude=0.005,f=None): #0.005 works
	wf=wavefront.copy()
	if f==None:
		m=calibrate_filter(f,num_modes,wf,wavefront_sensor,wavefront_estimator,amplitude)
	else:
		num_modes=len(dm.actuators)
		influence_functions = []
		
		wf.total_power = 1
		#lets get the proper lenslet measurements we want --- this should really be done by the wavefront sensor estimator
		img = wavefront_sensor(wf).intensity
		ref = wavefront_estimator.estimate([img]).ravel()
		influence_functions = []
		for i in range(num_modes):
			total_slopes = np.zeros(ref.shape)
			for amp in np.array([-1*amplitude,amplitude]):
		   
				act_levels = np.zeros(num_modes)
				act_levels[i] = (wf.wavelength) * amp
	
				dm.actuators = act_levels
				dm_wf = dm(wf)
				wfs = wavefront_sensor( dm_wf )
				wfs_img = wfs.intensity
		 
				slopes = wavefront_estimator.estimate([wfs_img]).ravel()
				total_slopes += (slopes- ref)/(2*amp)
			influence_functions.append(f.filter(total_slopes,0))
		influence_functions = ModeBasis(influence_functions)
		m = inverse_truncated(influence_functions.transformation_matrix)
	controller.set_interaction_matrix(m,0)