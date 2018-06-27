import numpy as np
import matplotlib.pyplot as plt
from ..field import make_pupil_grid
from ..optics import DeformableMirror
from ..mode_basis import ModeBasis
from ..math_util import inverse_truncated

class Modal_Filter(object):
	def __init__(self, mode_basis):
		
		if not hasattr(mode_basis, '__iter__'):
			self.mode_basis= [mode_basis]
		else:
			self.mode_basis = mode_basis
		self.flr=None
	def filter(self, wavefront,t,filter_number=0):
		fltr_wavefront = self.flr.dot(wavefront)
		return fltr_wavefront
	def set_filter(self,value,t):
		self.flr=value

def calibrate_filter(f,num_modes,wavefront,wavefront_sensor,wavefront_estimator,amplitude=0.005):
	wf=wavefront.copy()
	N=np.sqrt(wf.phase.shape)
	D=wf.grid.delta*N
	pupil_grid = make_pupil_grid(N,  1.4*D)
	#dm_modes = make_zernike_basis(num_modes, D, pupil_grid, 1, False)
	modes_freeform = DeformableMirror(f.mode_basis)
	wf.total_power = 1
	wf.electric_field=wf.electric_field*np.exp(-1j*0*wf.grid.x)
	#lets get the proper lenslet measurements we want --- this should really be done by the wavefront sensor estimator
	img = wavefront_sensor(wf).intensity
	ref = wavefront_estimator.estimate([img]).ravel()
	total=np.zeros(ref.shape)
	influence_functions = []

	for i in range(num_modes):
		total=np.zeros(ref.shape)
		for amp in np.array([-1*amplitude,amplitude]):  
			act_levels = np.zeros(num_modes)
			act_levels[i] = (wf.wavelength) * amp
			modes_freeform.actuators = act_levels
			dm_wf = modes_freeform(wf)
			wfs = wavefront_sensor( dm_wf )
			wfs_img = wfs.intensity
			
			slopes = wavefront_estimator.estimate([wfs_img]).ravel()
			total =total+( (slopes- ref)/(2*amp))
		influence_functions.append(total)
	influence_functions = ModeBasis(influence_functions)
	actuation_matrix = inverse_truncated(influence_functions.transformation_matrix)
	if f!=None:
		f.set_filter(actuation_matrix,0)
	return actuation_matrix