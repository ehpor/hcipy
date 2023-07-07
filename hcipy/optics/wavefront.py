import copy
import numpy as np
import numexpr as ne

from ..field import Field, field_dot, field_kron

_U_matrix = 1 / np.sqrt(2) * np.array([
    [1, 0, 0, 1],
    [1, 0, 0, -1],
    [0, 1, 1, 0],
    [0, 1j, -1j, 0]])

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
    input_stokes_vector : ndarray or None
        If a Stokes vector (I, Q, U, V) is given, a partially-polarized
        wavefront is initialized. If `electric_field` is scalar, it will
        be transformed into a tensor field with the correct Jones states.
        If a tensor-field is given as the `electric_field`, the electric
        field will be interpreted as the Jones matrix modifying the input
        Stokes vector.

    Raises
    ------
    ValueError
        When a Stokes vector is supplied but a vector field is given as
        electric field, or when an input Stokes vector is not supplied,
        but a 2-tensor field is given as electric field.
    '''
    def __init__(self, electric_field, wavelength=1, input_stokes_vector=None):
        if input_stokes_vector is not None:
            if electric_field.tensor_order not in [0, 2]:
                raise ValueError('When supplying a Stokes vector, the electric field must be either a scalar or 2-tensor field.')

            if electric_field.is_scalar_field:
                self.electric_field = electric_field[np.newaxis, np.newaxis, :] * np.eye(2)[..., np.newaxis]
            else:
                self.electric_field = electric_field

            self._input_stokes_vector = np.array(input_stokes_vector)
        else:
            self.electric_field = electric_field
            self._input_stokes_vector = None

            if electric_field.tensor_order == 2:
                raise ValueError('When supplying a 2-tensor field as electric field, an input Stokes vector is required.')

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
    def electric_field(self, electric_field):
        if not hasattr(electric_field, 'grid'):
            raise ValueError('The electric field must be a Field.')

        # Cast to complex with correct bit depth
        if electric_field.dtype == 'float32' or electric_field.dtype == 'complex64':
            dtype = 'complex64'
        else:
            dtype = 'complex128'

        self._electric_field = electric_field.astype(dtype, copy=False)

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
    def I(self):  # noqa: N802
        '''The I-component of the Stokes vector as function of 2D position
        in the plane.
        '''
        if self.is_scalar:
            # This is a scaler field.
            intensity = ne.evaluate('real(abs(elec))**2', local_dict={'elec': self.electric_field})
            return Field(intensity, self.electric_field.grid)
        elif self.is_partially_polarized:
            # This is a tensor field.
            x = self._electric_field[0, 0, :]
            y = self._electric_field[0, 1, :]
            z = self._electric_field[1, 0, :]
            w = self._electric_field[1, 1, :]

            a, b, c, d = self._input_stokes_vector

            # NumExpr is not smart enough to have abs(x) be real, so we have to take the real
            # part manually. This may change in later releases of NumExpr.
            M11 = 'real(abs(x))**2 + real(abs(y))**2 + real(abs(z))**2 + real(abs(w))**2'
            M12 = 'real(abs(x))**2 - real(abs(y))**2 + real(abs(z))**2 - real(abs(w))**2'
            M13 = '2 * (real(x) * real(y) + imag(x) * imag(y) + real(z) * real(w) + imag(z) * imag(w))'
            M14 = '2 * (-real(x) * imag(y) + imag(x) * real(y) - real(z) * imag(w) + imag(z) * real(w))'

            res = '0.5 * ((' + M11 + ') * a + (' + M12 + ') * b + (' + M13 + ') * c +  (' + M14 + ') * d)'
            local_dict = {'x': x, 'y': y, 'z': z, 'w': w, 'a': a, 'b': b, 'c': c, 'd': d}

            return Field(ne.evaluate(res, local_dict=local_dict), self.electric_field.grid)
        else:
            # This is a vector field.
            return np.sum(np.abs(self.electric_field)**2, axis=0)

    @property
    def Q(self):  # noqa: N802
        '''The Q-component of the Stokes vector as function of 2D position
        in the plane.
        '''
        if self.is_scalar:
            # This is a scaler field.
            return self.grid.zeros()
        elif self.is_partially_polarized:
            # This is a tensor field.
            x = self._electric_field[0, 0, :]
            y = self._electric_field[0, 1, :]
            z = self._electric_field[1, 0, :]
            w = self._electric_field[1, 1, :]

            a, b, c, d = self._input_stokes_vector

            M21 = 'real(abs(x))**2 + real(abs(y))**2 - real(abs(z))**2 - real(abs(w))**2'
            M22 = 'real(abs(x))**2 - real(abs(y))**2 - real(abs(z))**2 + real(abs(w))**2'
            M23 = '2 * (real(x) * real(y) + imag(x) * imag(y) - real(z) * real(w) - imag(z) * imag(w))'
            M24 = '2 * (-real(x) * imag(y) + imag(x) * real(y) + real(z) * imag(w) - imag(z) * real(w))'

            res = '0.5 * ((' + M21 + ') * a + (' + M22 + ') * b + (' + M23 + ') * c +  (' + M24 + ') * d)'
            local_dict = {'x': x, 'y': y, 'z': z, 'w': w, 'a': a, 'b': b, 'c': c, 'd': d}

            return Field(ne.evaluate(res, local_dict=local_dict), self.electric_field.grid)
        else:
            # This is a vector field.
            return np.abs(self.electric_field[0, :])**2 - np.abs(self.electric_field[1, :])**2

    @property
    def U(self):  # noqa: N802
        '''The U-component of the Stokes vector as function of 2D position
        in the plane.
        '''
        if self.is_scalar:
            # This is a scaler field.
            return self.grid.zeros()
        elif self.is_partially_polarized:
            # This is a tensor field.
            x = self._electric_field[0, 0, :]
            y = self._electric_field[0, 1, :]
            z = self._electric_field[1, 0, :]
            w = self._electric_field[1, 1, :]

            a, b, c, d = self._input_stokes_vector

            M31 = '2 * (real(x) * real(z) + imag(x) * imag(z) + real(y) * real(w) + imag(y) * imag(w))'
            M32 = '2 * (real(x) * real(z) + imag(x) * imag(z) - real(y) * real(w) - imag(y) * imag(w))'
            M33 = '2 * (real(x) * real(w) + imag(x) * imag(w) + real(y) * real(z) + imag(y) * imag(z))'
            M34 = '2 * (imag(x) * real(w) - real(x) * imag(w) + real(y) * imag(z) - imag(y) * real(z))'

            res = '0.5 * ((' + M31 + ') * a + (' + M32 + ') * b + (' + M33 + ') * c +  (' + M34 + ') * d)'
            local_dict = {'x': x, 'y': y, 'z': z, 'w': w, 'a': a, 'b': b, 'c': c, 'd': d}

            return Field(ne.evaluate(res, local_dict=local_dict), self.electric_field.grid)
        else:
            # This is a vector field.
            return 2 * np.real(self.electric_field[0, :] * self.electric_field[1, :].conj())

    @property
    def V(self):  # noqa: N802
        '''The V-component of the Stokes vector as function of 2D position
        in the plane.
        '''
        if self.is_scalar:
            # This is a scaler field.
            return self.grid.zeros()
        elif self.is_partially_polarized:
            # This is a tensor field.
            x = self._electric_field[0, 0, :]
            y = self._electric_field[0, 1, :]
            z = self._electric_field[1, 0, :]
            w = self._electric_field[1, 1, :]

            a, b, c, d = self._input_stokes_vector

            M41 = '2 * (real(x) * imag(z) - imag(x) * real(z) + real(y) * imag(w) - imag(y) * real(w))'
            M42 = '2 * (real(x) * imag(z) - imag(x) * real(z) - real(y) * imag(w) + imag(y) * real(w))'
            M43 = '2 * (real(x) * imag(w) - imag(x) * real(w) + real(y) * imag(z) - imag(y) * real(z))'
            M44 = '2 * (real(x) * real(w) + imag(x) * imag(w) - real(y) * real(z) - imag(y) * imag(z))'

            res = '0.5 * ((' + M41 + ') * a + (' + M42 + ') * b + (' + M43 + ') * c +  (' + M44 + ') * d)'
            local_dict = {'x': x, 'y': y, 'z': z, 'w': w, 'a': a, 'b': b, 'c': c, 'd': d}
            return Field(ne.evaluate(res, local_dict=local_dict), self.electric_field.grid)
        else:
            # This is a vector field.
            return -2 * np.imag(self.electric_field[0, :] * self.electric_field[1, :].conj())

    @property
    def stokes_vector(self):
        '''The Stokes vector.
        '''
        if self.is_scalar:
            # This is a scalar field and thus we return an unpolarized Stokes vector.
            stokes_vector = Field(np.zeros((4, self.grid.size)), self.grid)
            stokes_vector[0, :] = np.abs(self.electric_field)**2

            return stokes_vector
        elif self.is_partially_polarized:
            # This is a tensor field.
            mueller_matrix = jones_to_mueller(self.electric_field)

            return field_dot(mueller_matrix, self._input_stokes_vector)
        else:
            # This is a vector field and thus we return a fully polarized Stokes vector.
            stokes_vector = Field(np.zeros((4, self.grid.size)), self.grid)

            stokes_vector[0, :] = self.I
            stokes_vector[1, :] = self.Q
            stokes_vector[2, :] = self.U
            stokes_vector[3, :] = self.V

            return stokes_vector

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
        return 0.5 * np.arctan2(self.U, self.Q)

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
        if self.electric_field.is_scalar_field or self.electric_field.is_vector_field:
            variables = {'field': self.electric_field, 'weights': self.grid.weights}
            power = ne.evaluate('real(abs(field))**2 * weights', local_dict=variables)

            return Field(power, self.grid)
        else:
            return self.intensity * self.grid.weights

    @property
    def total_power(self):
        '''The total power in this wavefront.
        '''
        return np.sum(self.power)

    @total_power.setter
    def total_power(self, p):
        self.electric_field *= np.sqrt(p / self.total_power)

def jones_to_mueller(jones_matrix):
    '''Convert a Jones matrix to a Mueller matrix.

    Parameters
    ----------
    jones_matrix : ndarray or tensor field
        The Jones matrix/matrices to convert to Mueller matrix/matrices.

    Returns
    -------
    ndarray or tensor Field
        The Mueller matrix/matrices.
    '''
    return np.real(field_dot(_U_matrix, field_dot(field_kron(jones_matrix, jones_matrix.conj()), _U_matrix.conj().T)))
