import numpy as np
from .wavefront import Wavefront, jones_to_mueller
from .optical_element import OpticalElement, make_agnostic_optical_element
from ..field import Field, field_inv, field_dot, field_transpose, field_conjugate_transpose

def rotation_matrix(angle):
	'''Two dimensional rotation matrix. 

	Parameters
	----------
	angle : scaler
		rotation angle in radians 

	Returns
	-------
	ndarray 
		The rotation matrix.
	'''
	return np.array([[np.cos(angle),  np.sin(angle)],[-np.sin(angle), np.cos(angle)]])

@make_agnostic_optical_element([], ['jones_matrix'])
class JonesMatrixOpticalElement(OpticalElement):
	'''A general Jones Matrix.

	Parameters
	----------
	jones_matrix : matrix or Field
		The Jones matrix describing the optical element. 

	'''
	def __init__(self, jones_matrix, wavelength=1):
		self.jones_matrix = jones_matrix

	def forward(self, wavefront):
		'''Propgate the wavefront through the Jones matrix.

		Parameters
		----------
		wavefront : Wavefront
			The wavefront to propagate.
		
		Returns
		-------
		Wavefront
			The propagated wavefront.
		'''
		if wavefront.is_scalar:
			# we generate an unpolarized wavefront
			wf = Wavefront(wavefront.electric_field.copy(), wavelength=wavefront.wavelength, input_stokes_vector=[1,0,0,0])			
		else:
			wf = wavefront.copy()
		
		wf.electric_field = field_dot(self.jones_matrix, wf.electric_field)

		return wf
	
	def backward(self, wavefront):
		'''Propagate the wavefront backwards through the Jones matrix.

		Parameters
		----------
		wavefront : Wavefront
			The wavefront to propagate.
		
		Returns
		-------
		Wavefront
			The propagated wavefront.
		'''
		if wavefront.is_scalar:
			# we generate an unpolarized wavefront
			wf = Wavefront(wavefront.electric_field.copy(), wavelength=wavefront.wavelength, input_stokes_vector=[1,0,0,0])			
		else:
			wf = wavefront.copy()

		wf.electric_field = field_dot(field_conjugate_transpose(self.jones_matrix), wf.electric_field)

		return wf

	@property
	def mueller_matrix(self):
		'''Returns the Mueller matrix of the Jones matrix. 
		'''
		return jones_to_mueller(self.jones_matrix)

	def __mul__(self, m):
		'''Multiply the Jones matrix on the right side with the Jones matrix m. 

		Parameters
		----------
		m : JonesMatrixOpticalElement, Field or matrix
			The Jones matrix with which to multiply. 
		'''
		if hasattr(m, 'jones_matrix'):
			return JonesMatrixOpticalElement(field_dot(self.jones_matrix, m.jones_matrix))
		else: 
			return JonesMatrixOpticalElement(field_dot(self.jones_matrix, m))

@make_agnostic_optical_element([], ['phase_retardation', 'fast_axis_orientation', 'circularity'])
class PhaseRetarder(JonesMatrixOpticalElement):
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

		# calculating the individual components 
		j11 = phi_plus * np.cos(fast_axis_orientation)**2 + phi_minus * np.sin(fast_axis_orientation)**2
		j12 = (phi_plus - phi_minus) * np.exp(-1j * circularity) * np.cos(fast_axis_orientation) * np.sin(fast_axis_orientation)
		j21 = (phi_plus - phi_minus) * np.exp(1j * circularity) * np.cos(fast_axis_orientation) * np.sin(fast_axis_orientation)
		j22 = phi_plus * np.sin(fast_axis_orientation)**2 + phi_minus * np.cos(fast_axis_orientation)**2

		# constructing the Jones matrix. 
		jones_matrix = np.array([[j11, j12],[j21, j22]])

		# Testing if we can make it a Field. 
		if hasattr(j11, 'grid'):
			jones_matrix = Field(jones_matrix, j11.grid)

		JonesMatrixOpticalElement.__init__(self, jones_matrix, wavelength)

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

class GeometricPhaseElement(LinearRetarder):
	'''A general geometric phase element. 

	Parameters
	----------
	phase_pattern : Field or array_like 
		The phase pattern in radians. 
	leakage : scalar 
		The relative leakage strength (0 = no leakage, 1 = maximum leakage)
	retardance_offset : scalar 
		The retardance offset from half wave in radians. This will result in leakage.
	'''
	def __init__(self, phase_pattern, leakage=None, retardance_offset=0, wavelength=1):

		if leakage is not None:
			# Testing if the leakage is valid.
			if leakage < 0 or leakage > 1:
				raise ValueError('Leakage must be between 0 and 1.')

			# calculating the required retardance offset to get the leakage. 
			retardance_offset = 2 * np.arcsin(np.sqrt(leakage))

		LinearRetarder.__init__(self, np.pi-retardance_offset, phase_pattern/2, wavelength)

class LinearPolarizer(JonesMatrixOpticalElement):
	'''A linear polarizer.

	Parameters
	----------
	polarization_angle : scalar or Field
		The polarization angle of the polarizer. Light with this angle is transmitted.
	'''
	def __init__(self, polarization_angle, wavelength=1):
		self.polarization_angle = polarization_angle

		JonesMatrixOpticalElement.__init__(self, self.jones_matrix, wavelength)
	
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

		self.jones_matrix = np.array([[c**2, c * s],[c * s, s**2]])

class RotatedJonesMatrixOpticalElement(JonesMatrixOpticalElement):
	'''An axial rotated Jones matrix.

	Note: this rotates the polarization state, not the Field!

	Parameters
	----------
	jones_matrix : tensor field
		The Jones matrix that will be rotated.
	angle : scalar 
		The rotation angle in radians. 
	'''
	def __init__(self, jones_matrix, angle, wavelength=1):
		
		# calculating the rotated Jones matrix.
		rotated_jones_matrix = field_dot(rotation_matrix(-angle), field_dot(jones_matrix, rotation_matrix(angle)))

		JonesMatrixOpticalElement.__init__(self, rotated_jones_matrix, wavelength)

class PolarizingBeamSplitter(OpticalElement):
	''' A polarizing beam splitter that accepts one wavefront and returns two polarized wavefronts.

	The first of the returned wavefront will have the polarization state set by the polarization angle. The second wavefront
	 will be perpendicular to the first.

	Parameters
	----------
	polarization_angle : scalar or Field
		The polarization angle of the polarizer. 
	'''
	def __init__(self, polarization_angle, wavelength=1):
		self.polarization_angle = polarization_angle
		
	@property
	def polarization_angle(self):
		'''The angle of polarization of the linear polarizer.
		'''
		return self._polarization_angle
	
	@polarization_angle.setter
	def polarization_angle(self, polarization_angle):
		self._polarization_angle = polarization_angle

		# calculating two Jones matrices for the two ports.		
		self.polarizer_port_1 = LinearPolarizer(polarization_angle)
		self.polarizer_port_2 = LinearPolarizer(polarization_angle + np.pi / 2)

	def forward(self, wavefront):
		'''Propgate the wavefront through the PBS.

		Parameters
		----------
		wavefront : Wavefront
			The wavefront to propagate.
		
		Returns
		-------
		wf_1 : Wavefront 
			The wavefront propagated through a polarizer under the polarization angle.

		wf_2 : Wavefront 
			The propagated wavefront through a polarizer perpendicular to the first one.
		'''

		wf_1 = self.polarizer_port_1.forward(wavefront)
		wf_2 = self.polarizer_port_2.forward(wavefront)

		return wf_1, wf_2

	def backward(self, wavefront):
		'''Propagate the wavefront backwards through the PBS.

		Not possible, will raise error.
		'''
		raise RuntimeError('Backward propagation through PolarizingBeamSplitter not possible.')

	@property
	def mueller_matrix(self):
		'''Returns the Mueller matrices of the two Jones matrices.
		'''
		return jones_to_mueller(self.polarizer_port_1.jones_matrix), jones_to_mueller(self.polarizer_port_2.jones_matrix)

#class Reflection(JonesMatrixOpticalElement):
''' A jones matrix that handles the flips in polarization for a reflection. + field flip around x- or y-axis 
'''
