import numpy as np
from .optical_element import OpticalElement, make_polychromatic
from ..field import Field, field_inv

class PhaseRetarderMonochromatic(object):
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
		wf = wavefront.copy()
		wf.electric_field = field_dot(wf.electric_field, self.jones_matrix)
		return wf
	
	def backward(self, wavefront):
		wf = wavefront.copy()
		wf.electric_field = field_dot(wf.electric_field, self.jones_matrix_inv)
		return wf

PhaseRetarder = make_polychromatic(["phase_retardation", "fast_axis_orientation", "circularity"])(PhaseRetarderMonochromatic)

class LinearRetarderMonochromatic(PhaseRetarderMonochromatic):
	def __init__(self, phase_retardation, fast_axis_orientation, wavelength=1):
		PhaseRetarderMonochromatic.__init__(self, phase_retardation, fast_axis_orientation, 0, wavelength)

LinearRetarder = make_polychromatic(["phase_retardation", "fast_axis_orientation"])(LinearRetarderMonochromatic)

class CircularRetarderMonochromatic(PhaseRetarderMonochromatic):
	def __init__(self, phase_retardation, wavelength=1):
		PhaseRetarderMonochromatic.__init__(self, phase_retardation, np.pi / 4, np.pi / 2, wavelength)

CircularRetarder = make_polychromatic(["phase_retardation"])(CircularRetarderMonochromatic)

class QuarterWavePlateMonochromatic(LinearRetarderMonochromatic):
	def __init__(self, fast_axis_orientation, wavelength=1):
		LinearRetarderMonochromatic.__init__(self, np.pi / 2, fast_axis_orientation, wavelength)

QuarterWavePlate = make_polychromatic(["fast_axis_orientation"])(QuarterWavePlateMonochromatic)

class HalfWavePlateMonochromatic(LinearRetarderMonochromatic):
	def __init__(self, fast_axis_orientation, wavelength=1):
		LinearRetarderMonochromatic.__init__(self, np.pi, fast_axis_orientation, wavelength)

HalfWavePlate = make_polychromatic(["fast_axis_orientation"])(HalfWavePlateMonochromatic)

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