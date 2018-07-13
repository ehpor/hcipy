import numpy as np
import matplotlib.pyplot as plt

from ..field import make_pupil_grid
from ..optics import DeformableMirror
from ..mode_basis import ModeBasis
from ..math_util import inverse_truncated


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