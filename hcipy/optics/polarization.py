import numpy as np
from .optical_element import OpticalElement, make_agnostic_optical_element
from ..field import Field, field_inv, field_dot, field_transpose, field_conjugate_transpose

def tensor_product(M1, M2):
    '''
    This function calculates the tensor product as defined in:
    https://en.wikipedia.org/wiki/Tensor_product,
    to make from 2 2x2 tensor fields 1 4x4 tensor field.
    M1 x M2= result
    Assuming the M1 and M2 are 2x2 tensor fields.
    '''
    result = Field(np.zeros((4, 4, M1.shape[-1]), dtype = np.complex128), M1.grid)

    #Performing the tensor product.
    result[0:2,0:2,:] += np.copy(M1[0,0,:] * M2)
    result[0:2,2:4,:] += np.copy(M1[0,1,:] * M2)
    result[2:4,0:2,:] += np.copy(M1[1,0,:] * M2)
    result[2:4,2:4,:] += np.copy(M1[1,1,:] * M2)

    return result

def jones_to_mueller(jones_matrix):

    #The Jones/Mueller transformation matrix.
    U = jones_matrix.grid.zeros((4,4), 'complex')

    U[0,0,:] += 1
    U[0,3,:] += 1
    U[1,0,:] += 1
    U[1,3,:] += -1
    U[2,1,:] += 1
    U[2,2,:] += 1
    U[3,1,:] += 1j
    U[3,2,:] += -1j
    U *= 1/np.sqrt(2)

    #Now we also calculate the Hermitian adjoint of U to get the inverse.
    U_inverse = field_transpose(U)

    #Defining the conjugate of the Jones matrix.
    jones_matrix_conj = Field(np.zeros_like(jones_matrix), jones_matrix.grid)
    jones_matrix_conj += np.copy(jones_matrix).conj()

    #Taking the real part of the result, as that should be completely real. 
    return np.real(field_dot(U, field_dot(tensor_product(jones_matrix, jones_matrix_conj), U_inverse)))

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
		wf = wavefront.copy()
		wf.electric_field = field_dot(field_conjugate_transpose(self.jones_matrix), wf.electric_field)
		return wf

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

class LinearPolarizer(JonesMatrixOpticalElement):
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