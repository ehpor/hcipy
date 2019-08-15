import copy
import numpy as np

from ..field import Field, field_dot

# TODO Should add a pilot Gaussian beam with each Wavefront

class Wavefront(object):
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
		return copy.deepcopy(self)
	
	@property
	def electric_field(self):
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
		return self._input_stokes_vector

	@property
	def wavenumber(self):
		return 2 * np.pi / self.wavelength
	
	@wavenumber.setter
	def wavenumber(self, wavenumber):
		self.wavelength = 2 * np.pi / wavenumber
	
	@property
	def grid(self):
		return self.electric_field.grid

	@property
	def I(self):
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

	@property
	def Q(self):
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
	
	@property
	def U(self):
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

	@property
	def V(self):
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
	
	@property
	def degree_of_polarization(self):
		return np.sqrt(self.Q**2 + self.U**2 + self.V**2) / self.I

	@property
	def intensity(self):
		return np.abs(self.electric_field)**2

	@property
	def amplitude(self):
		return np.abs(self.electric_field)

	@property
	def phase(self):
		phase = np.angle(self.electric_field)
		return Field(phase, self.electric_field.grid)
	
	@property
	def real(self):
		return np.real(self.electric_field)
	
	@property
	def imag(self):
		return np.imag(self.electric_field)
	
	@property
	def power(self):
		return self.intensity * self.grid.weights
	
	@property
	def total_power(self):
		return np.sum(self.power)
	
	@total_power.setter
	def total_power(self, p):
		self.electric_field *= np.sqrt(p / self.total_power)