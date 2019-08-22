import copy
import numpy as np

from ..field import Field, field_dot

# TODO Should add a pilot Gaussian beam with each Wavefront

class Wavefront(object):
	'''A physical wavefront in an optical system.

	This represents the state of light to be propagated through the
	optical system. It can be both an electric field in the scalar 
	approximation (ie. scalar wavefront propgation), a fully polarized
	wavefront, represented by a Field of Jones vectors, and a potentially
	partially-polarized wavefront, represented by two Jones vector fields 
	and the Stokes vector corresponding to the Jones vectors (1,0) and (0,1).

	Parameters
	----------
	electric_field : Field
		The electric field, either scalar, vectorial or 2-tensorial.
	wavelength : scalar
		The wavelength of the wavefront.
	stokes_vector : ndarray or None
		If a Stokes vector (I, Q, U, V) is given, a partially-polarized
		wavefront is initialized. If `electric_field` is scalar, it will
		be transformed into a tensor field with the correct Jones states.
		If a tensor-field is given as the `electric_field`, the Stokes
		vector will be interpreted as corresponding to the Jones vectors
		(1,0) and (0,1).
	'''
	def __init__(self, electric_field, wavelength=1, stokes_vector=None):
		if stokes_vector is not None:
			if electric_field.tensor_order not in [0, 2]:
				raise ValueError('When supplying a Stokes vector, the electric field must be either a scalar or 2-tensor field.')
			
			if electric_field.is_scalar_field:
				self._electric_field = electric_field[np.newaxis, np.newaxis, :].astype('complex') * np.eye(2)[..., np.newaxis]
			else:
				self._electric_field = electric_field.astype('complex')
			
			self._input_stokes_vector = np.array(stokes_vector)
		else:
			self._electric_field = electric_field.astype('complex')
			self._input_stokes_vector = None

		self.wavelength = wavelength
	
	def copy(self):
		'''Make a copy of the wavefront.
		'''
		return copy.deepcopy(self)
	
	@property
	def electric_field(self):
		'''The electric field as function of 2D position on the plane.
		'''
		return self._electric_field
	
	@electric_field.setter
	def electric_field(self, U):
		if hasattr(U, 'grid'):
			self._electric_field = U.astype('complex')
		else:
			if len(U) == 2:
				self._electric_field = Field(U[0].astype('complex'), U[1])
			else:
				raise ValueError("Electric field requires an accompanying grid.")
	
	@property
	def input_stokes_vector(self):
		'''The Stokes vector corresponding to the Jones vectors (1,0) and (0,1).
		'''
		return self._input_stokes_vector

	@property
	def wavenumber(self):
		'''The wavenumber of the light.
		'''
		return 2 * np.pi / self.wavelength
	
	@wavenumber.setter
	def wavenumber(self, wavenumber):
		self.wavelength = 2 * np.pi / wavenumber
	
	@property
	def grid(self):
		'''The grid on which the electric field is defined.
		'''
		return self.electric_field.grid

	@property
	def I(self):
		'''The I-component of the Stokes vector as function of 2D position
		in the plane.
		'''

		if self.is_scalar:
			# This is a scaler field. 
			return np.abs(self.electric_field)**2
		elif self.is_partially_polarized:
			# This is a tensor field. 
			x = self.electric_field[0, 0, :]
			y = self.electric_field[0, 1, :]
			z = self.electric_field[1, 0, :]
			w = self.electric_field[1, 1, :]

			M11 = x * x.conj() + y * y.conj() + z * z.conj() + w * w.conj()
			M12 = x * x.conj() - y * y.conj() + z * z.conj() - w * w.conj()
			M13 = x * y.conj() + y * x.conj() + z * w.conj() + w * z.conj()
			M14 = 1j * (x * y.conj() - y * x.conj() + z * w.conj() - w * z.conj())

			row = Field(np.array([M11, M12, M13, M14]), self.electric_field.grid)
			
			return 0.5 * field_dot(row, self._input_stokes_vector).real
		else:
			# This is a vector field. 
			return np.sum(np.abs(self.electric_field)**2, axis=0)

	@property
	def Q(self):
		'''The Q-component of the Stokes vector as function of 2D position
		in the plane.
		'''
		if self.is_scalar:
			# This is a scaler field. 
			return self.grid.zeros()
		elif self.is_partially_polarized:
			# This is a tensor field. 
			x = self.electric_field[0, 0, :]
			y = self.electric_field[0, 1, :]
			z = self.electric_field[1, 0, :]
			w = self.electric_field[1, 1, :]

			M21 = x * x.conj() + y * y.conj() - z * z.conj() - w * w.conj()
			M22 = x * x.conj() - y * y.conj() - z * z.conj() + w * w.conj()
			M23 = x * y.conj() + y * x.conj() - z * w.conj() - w * z.conj()
			M24 = 1j * (x * y.conj() - y * x.conj() - z * w.conj() + w * z.conj())

			row = Field(np.array([M21, M22, M23, M24]), self.electric_field.grid)
			
			return 0.5 * field_dot(row, self._input_stokes_vector).real
		else:
			# This is a vector field. 
			return np.abs(self.electric_field[0,:])**2 - np.abs(self.electric_field[1,:])**2
	
	@property
	def U(self):
		'''The U-component of the Stokes vector as function of 2D position
		in the plane.
		'''
		if self.is_scalar:
			# This is a scaler field.
			return self.grid.zeros()
		elif self.is_partially_polarized:
			# This is a tensor field. 
			x = self.electric_field[0, 0, :]
			y = self.electric_field[0, 1, :]
			z = self.electric_field[1, 0, :]
			w = self.electric_field[1, 1, :]

			M31 = x * z.conj() + y * w.conj() + z * x.conj() + w * y.conj()
			M32 = x * z.conj() - y * w.conj() + z * x.conj() - w * y.conj()
			M33 = x * w.conj() + y * z.conj() + z * y.conj() + w * x.conj()
			M34 = 1j * (x * w.conj() - y * z.conj() + z * y.conj() - w * x.conj())
			
			row = Field(np.array([M31, M32, M33, M34]), self.electric_field.grid)
			
			return 0.5 * field_dot(row, self._input_stokes_vector).real
		else:
			# This is a vector field. 
			return 2 * np.real(self.electric_field[0,:] * self.electric_field[1,:].conj())

	@property
	def V(self):
		'''The V-component of the Stokes vector as function of 2D position
		in the plane.
		'''
		if self.is_scalar:
			# This is a scaler field.
			return self.grid.zeros()
		elif self.is_partially_polarized:
			# This is a tensor field. 
			x = self.electric_field[0, 0, :]
			y = self.electric_field[0, 1, :]
			z = self.electric_field[1, 0, :]
			w = self.electric_field[1, 1, :]

			M41 = 1j * (-x * z.conj() - y * w.conj() + z * x.conj() + w * y.conj())
			M42 = 1j * (-x * z.conj() + y * w.conj() + z * x.conj() - w * y.conj())
			M43 = 1j * (-x * w.conj() - y * z.conj() + z * y.conj() + w * x.conj())
			M44 = x * w.conj() - y * z.conj() - z * y.conj() + w * x.conj()
			
			row = Field(np.array([M41, M42, M43, M44]), self.electric_field.grid)
			
			return 0.5 * field_dot(row, self._input_stokes_vector).real
		else:
			# This is a vector field. 
			return -2 * np.imag(self.electric_field[0,:] * self.electric_field[1,:].conj())
	
	@property
	def degree_of_polarization(self):
		'''The degree of polarization.
		'''
		return np.sqrt(self.Q**2 + self.U**2 + self.V**2) / self.I
	
	@property
	def degree_of_linear_polarization(self):
		'''The degree of linear polarization.
		'''
		return np.sqrt(self.Q**2 + self.U**2) / self.I

	@property
	def angle_of_linear_polarization(self):
		'''The angle of linear polarization.
		'''
		return 0.5 * np.arctan(self.U / self.Q)
	
	@property
	def degree_of_circular_polarization(self):
		'''The degree of circular polarization.
		'''
		return self.V / self.I
	
	@property
	def ellipticity(self):
		'''The ratio of the minor to major axis of the electric
		field polarization ellipse.
		'''
		return self.V / (self.I + np.sqrt(self.Q**2 + self.U**2))
	
	@property
	def is_polarized(self):
		'''If the wavefront can be polarized.
		'''
		return self.electric_field.tensor_order in [1, 2]
	
	@property
	def is_partially_polarized(self):
		'''If the wavefront can be partially polarized.
		'''
		return self.electric_field.tensor_order == 2
	
	@property
	def is_scalar(self):
		'''If the wavefront uses the scalar approximation.
		'''
		return self.electric_field.is_scalar_field

	@property
	def intensity(self):
		'''The total intensity of the wavefront as function of 2D position on the plane.
		'''
		return self.I

	@property
	def amplitude(self):
		'''The amplitude of the wavefront as function of 2D position on the plane.
		'''
		return np.abs(self.electric_field)

	@property
	def phase(self):
		'''The phase of the wavefront as function of 2D position on the plane.
		'''
		phase = np.angle(self.electric_field)
		return Field(phase, self.electric_field.grid)
	
	@property
	def real(self):
		'''The real part of the wavefront as function of 2D position on the plane.
		'''
		return np.real(self.electric_field)
	
	@property
	def imag(self):
		'''The imaginary part of the wavefront as function of 2D position on the plane.
		'''
		return np.imag(self.electric_field)
	
	@property
	def power(self):
		'''The power of each pixel in the wavefront.
		'''
		return self.intensity * self.grid.weights
	
	@property
	def total_power(self):
		'''The total power in this wavefront.
		'''
		return np.sum(self.power)
	
	@total_power.setter
	def total_power(self, p):
		self.electric_field *= np.sqrt(p / self.total_power)
