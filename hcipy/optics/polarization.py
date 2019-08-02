import numpy as np
from .optical_element import OpticalElement, make_agnostic_optical_element
from ..field import Field, field_inv, field_dot

@make_agnostic_optical_element([], ['phase_retardation', 'fast_axis_orientation', 'circularity'])
class PhaseRetarder(OpticalElement):
	'''A general phase retarder.

	Parameters
	----------
	phase_retardation : scalar or Field
		The relative phase retardation induced between the fast and slow axis.
	fast_axis_orientation : scalar or Field
		The angle of the fast axis with respect to the x-axis in radians.
	circularity : scalar or Field
		The circularity of the phase retarder.
	'''
	def __init__(self, phase_retardation, fast_axis_orientation, circularity, wavelength=1):
		phi_plus = np.exp(1j * phase_retardation / 2)
		phi_minus = np.exp(-1j * phase_retardation / 2)

		j11 = phi_plus * np.cos(fast_axis_orientation)**2 + phi_minus * np.sin(fast_axis_orientation)**2
		j12 = (phi_plus - phi_minus) * np.exp(-1j * circularity) * np.cos(fast_axis_orientation) * np.sin(fast_axis_orientation)
		j21 = (phi_plus - phi_minus) * np.exp(1j * circularity) * np.cos(fast_axis_orientation) * np.sin(fast_axis_orientation)
		j22 = phi_plus * np.sin(fast_axis_orientation)**2 + phi_minus * np.cos(fast_axis_orientation)**2

		self.jones_matrix = np.array([[j11, j12],[j21, j22]])
		if hasattr(j11, 'grid'):
			self.jones_matrix = Field(self.jones_matrix, j11.grid)

		self.jones_matrix_inv = field_inv(self.jones_matrix)
	
	def forward(self, wavefront):
		'''Propgate the wavefront through the phase retarder.

		Parameters
		----------
		wavefront : Wavefront
			The wavefront to propagate.
		
		Returns
		-------
		Wavefront
			The propagated wavefront.
		'''
		wf = wavefront.copy()
		wf.electric_field = field_dot(wf.electric_field, self.jones_matrix)
		return wf
	
	def backward(self, wavefront):
		'''Propagate the wavefront backwards through the phase retarder.

		Parameters
		----------
		wavefront : Wavefront
			The wavefront to propagate.
		
		Returns
		-------
		Wavefront
			The propagated wavefront.
		'''
		wf = wavefront.copy()
		wf.electric_field = field_dot(wf.electric_field, self.jones_matrix_inv)
		return wf

class LinearRetarder(PhaseRetarder):
	'''A general linear retarder.

	Parameters
	----------
	phase_retardation : scalar or Field
		The relative phase retardation induced between the fast and slow axis.
	fast_axis_orientation : scalar or Field
		The angle of the fast axis with respect to the x-axis in radians.
	'''
	def __init__(self, phase_retardation, fast_axis_orientation, wavelength=1):
		PhaseRetarder.__init__(self, phase_retardation, fast_axis_orientation, 0, wavelength)

class CircularRetarder(PhaseRetarder):
	'''A general circular retarder.
	
	Parameters
	----------
	phase_retardation : scalar or Field
		The relative phase retardation induced between the fast and slow axis.
	fast_axis_orientation : scalar or Field
		The angle of the fast axis with respect to the x-axis in radians.
	'''
	def __init__(self, phase_retardation, wavelength=1):
		PhaseRetarder.__init__(self, phase_retardation, np.pi / 4, np.pi / 2, wavelength)

class QuarterWavePlate(LinearRetarder):
	'''A quarter-wave plate.
	
	Parameters
	----------
	fast_axis_orientation : scalar or Field
		The angle of the fast axis with respect to the x-axis in radians.
	'''
	def __init__(self, fast_axis_orientation, wavelength=1):
		LinearRetarder.__init__(self, np.pi / 2, fast_axis_orientation, wavelength)

class HalfWavePlate(LinearRetarder):
	'''A half-wave plate.
	
	Parameters
	----------
	fast_axis_orientation : scalar or Field
		The angle of the fast axis with respect to the x-axis in radians.
	'''
	def __init__(self, fast_axis_orientation, wavelength=1):
		LinearRetarder.__init__(self, np.pi, fast_axis_orientation, wavelength)

class LinearPolarizer(OpticalElement):
	'''A linear polarizer.

	Parameters
	----------
	polarization_angle : scalar or Field
		The polarization angle of the polarizer. Light with this angle is transmitted.
	'''
	def __init__(self, polarization_angle):
		self.polarization_angle = polarization_angle
	
	@property
	def polarization_angle(self):
		'''The angle of polarization of the linear polarizer.
		'''
		return self._polarization_angle
	
	@polarization_angle.setter
	def polarization_angle(self, polarization_angle):
		self._polarization_angle = polarization_angle

		c = np.cos(polarization_angle)
		s = np.sin(polarization_angle)
		self.jones_matrix = np.array([[c**2, -c * s],[-c * s, s**2]])
	
	def forward(self, wavefront):
		'''Propagate a wavefront forwards through the linear polarizer.

		Parameters
		----------
		wavefront : Wavefront
			The wavefront to propagate.
		
		Returns
		-------
		Wavefront
			The propagated wavefront.
		'''
		wf = wavefront.copy()
		wf.electric_field = field_dot(self.jones_matrix, wavefront.electric_field)
		return wf
	
	def backward(self, wavefront):
		'''Propagate a wavefront backwards through the linear polarizer.

		Parameters
		----------
		wavefront : Wavefront
			The wavefront to propagate.
		
		Returns
		-------
		Wavefront
			The propagated wavefront.
		'''
		return self.forward(wavefront)