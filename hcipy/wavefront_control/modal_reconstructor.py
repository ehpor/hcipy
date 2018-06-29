import numpy as np
import matplotlib.pyplot as plt

from ..field import make_pupil_grid
from ..optics import DeformableMirror
from ..mode_basis import ModeBasis
from ..math_util import inverse_truncated
from .observer import Observer

class ModalReconstructor(Observer):
	def __init__(self, mode_basis):
		if not hasattr(mode_basis, '__iter__'):
			self.mode_basis = [mode_basis]
		else:
			self.mode_basis = mode_basis
		
		self.flr = None
	
	def estimate(self, wavefront, t, filter_number=0):
		return self.flr.dot(wavefront)
	
	def set_filter(self,value,t):
		self.flr = value

def calibrate_modal_reconstructor(f, num_modes, wavefront, wavefront_sensor, wavefront_estimator, amplitude=0.005):
	wf = wavefront.copy()
	
	N = wf.electric_field.grid.shape
	D = max(np.ptp(wf.electric_field.grid.x), np.ptp(wf.electric_field.grid.y))
	pupil_grid = make_pupil_grid(N, 1.4 * D)
	
	modes_freeform = DeformableMirror(f.mode_basis)
	wf.total_power = 1
	wf.electric_field *= np.exp(-1j*0*wf.grid.x)

	# Let's get the proper lenslet measurements we want. This should really be done by the wavefront sensor estimator
	img = wavefront_sensor(wf).intensity
	ref = wavefront_estimator.estimate([img]).ravel()
	total = np.zeros(ref.shape)
	influence_functions = []

	for i in range(num_modes):
		total = np.zeros(ref.shape)

		for amp in np.array([-amplitude, amplitude]):
			act_levels = np.zeros(num_modes)
			act_levels[i] = wf.wavelength * amp
			modes_freeform.actuators = act_levels

			dm_wf = modes_freeform(wf)
			wfs = wavefront_sensor(dm_wf)
			wfs_img = wfs.intensity
			
			slopes = wavefront_estimator.estimate([wfs_img]).ravel()
			total = total + (slopes - ref) / (2 * amp)
		influence_functions.append(total)
	
	influence_functions = ModeBasis(influence_functions)
	actuation_matrix = inverse_truncated(influence_functions.transformation_matrix)

	if f is not None:
		f.set_filter(actuation_matrix, 0)
	
	return actuation_matrix