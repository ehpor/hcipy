import copy
import math

from ..field import Field, NewStyleField, field_dot, is_field
from .._math import optics as optics_math
from .._math.backends import array_namespace, complex_dtype

_U_MATRIX = tuple(
    tuple(1 / math.sqrt(2) * value for value in row)
    for row in (
        (1, 0, 0, 1),
        (1, 0, 0, -1),
        (0, 1, 1, 0),
        (0, 1j, -1j, 0),
    )
)

_U_MATRIX_CONJUGATE_TRANSPOSE = tuple(
    tuple(value.conjugate() for value in column)
    for column in zip(*_U_MATRIX)
)

# TODO Should add a pilot Gaussian beam with each Wavefront

def _field_data(field):
    return field.data if isinstance(field, NewStyleField) else field

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
                xp = array_namespace(electric_field)
                dtype = complex_dtype(xp, electric_field.dtype)
                expanded_field = electric_field[None, None, :] * xp.eye(2, dtype=dtype)[..., None]
                self.electric_field = expanded_field
            else:
                self.electric_field = electric_field

            self._input_stokes_vector = electric_field.grid.xp.asarray(input_stokes_vector)
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

        xp = array_namespace(electric_field)
        dtype = complex_dtype(xp, electric_field.dtype)

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
        return 2 * math.pi / self.wavelength

    @wavenumber.setter
    def wavenumber(self, wavenumber):
        self.wavelength = 2 * math.pi / wavenumber

    @property
    def grid(self):
        '''The grid on which the electric field is defined.
        '''
        return self.electric_field.grid

    @property
    def I(self):  # noqa: N802, E743
        '''The I-component of the Stokes vector as function of 2D position
        in the plane.
        '''
        result = optics_math.stokes_i(_field_data(self.electric_field), self._input_stokes_vector)

        return Field(result, self.grid)

    @property
    def Q(self):  # noqa: N802
        '''The Q-component of the Stokes vector as function of 2D position
        in the plane.
        '''
        result = optics_math.stokes_q(_field_data(self.electric_field), self._input_stokes_vector)

        return Field(result, self.grid)

    @property
    def U(self):  # noqa: N802
        '''The U-component of the Stokes vector as function of 2D position
        in the plane.
        '''
        result = optics_math.stokes_u(_field_data(self.electric_field), self._input_stokes_vector)

        return Field(result, self.grid)

    @property
    def V(self):  # noqa: N802
        '''The V-component of the Stokes vector as function of 2D position
        in the plane.
        '''
        result = optics_math.stokes_v(_field_data(self.electric_field), self._input_stokes_vector)

        return Field(result, self.grid)

    @property
    def stokes_vector(self):
        '''The Stokes vector.
        '''
        if self.is_partially_polarized:
            return field_dot(jones_to_mueller(self.electric_field), self._input_stokes_vector)

        data = _field_data(self.electric_field)
        xp = array_namespace(data)
        result = xp.stack([
            optics_math.stokes_i(data, self._input_stokes_vector),
            optics_math.stokes_q(data, self._input_stokes_vector),
            optics_math.stokes_u(data, self._input_stokes_vector),
            optics_math.stokes_v(data, self._input_stokes_vector),
        ], axis=0)

        return Field(result, self.grid)

    @property
    def degree_of_polarization(self):
        '''The degree of polarization.
        '''
        result = optics_math.polarization_degree(
            _field_data(self.I),
            _field_data(self.Q),
            _field_data(self.U),
            _field_data(self.V),
        )

        return Field(result, self.grid)

    @property
    def degree_of_linear_polarization(self):
        '''The degree of linear polarization.
        '''
        result = optics_math.linear_polarization_degree(
            _field_data(self.I),
            _field_data(self.Q),
            _field_data(self.U),
        )

        return Field(result, self.grid)

    @property
    def angle_of_linear_polarization(self):
        '''The angle of linear polarization.
        '''
        result = optics_math.linear_polarization_angle(_field_data(self.Q), _field_data(self.U))

        return Field(result, self.grid)

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
        result = optics_math.ellipticity(
            _field_data(self.I),
            _field_data(self.Q),
            _field_data(self.U),
            _field_data(self.V),
        )

        return Field(result, self.grid)

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
        xp = array_namespace(self.electric_field)

        return xp.abs(self.electric_field)

    @property
    def phase(self):
        '''The phase of the wavefront as function of 2D position on the plane.
        '''
        result = optics_math.phase(_field_data(self.electric_field))

        return Field(result, self.grid)

    @property
    def real(self):
        '''The real part of the wavefront as function of 2D position on the plane.
        '''
        xp = array_namespace(self.electric_field)

        return xp.real(self.electric_field)

    @property
    def imag(self):
        '''The imaginary part of the wavefront as function of 2D position on the plane.
        '''
        xp = array_namespace(self.electric_field)

        return xp.imag(self.electric_field)

    @property
    def power(self):
        '''The power of each pixel in the wavefront.
        '''
        result = optics_math.power(
            _field_data(self.electric_field),
            self.grid.weights,
            self._input_stokes_vector,
        )

        return Field(result, self.grid)

    @property
    def total_power(self):
        '''The total power in this wavefront.
        '''
        xp = array_namespace(self.power)

        return xp.sum(self.power)

    @total_power.setter
    def total_power(self, p):
        xp = array_namespace(self.electric_field)

        scale = xp.astype(xp.sqrt(p / self.total_power), self.electric_field.dtype)
        self.electric_field = self.electric_field * scale

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
    xp = array_namespace(jones_matrix)

    data = xp.asarray(jones_matrix)
    data_conjugate = xp.conj(data)

    transformation_matrix = xp.asarray(_U_MATRIX)
    transformation_matrix_conjugate = xp.asarray(_U_MATRIX_CONJUGATE_TRANSPOSE)

    if is_field(jones_matrix):
        # Kronecker product of the Jones matrix with its conjugate.
        n = data.shape[-1]
        kron = data[:, None, :, None, :] * data_conjugate[None, :, None, :, :]
        kron = xp.reshape(kron, (4, 4, n))
        kron = xp.permute_dims(kron, (2, 0, 1))

        # Transform the Kronecker product into Mueller matrices.
        mueller_matrix = xp.matmul(xp.matmul(transformation_matrix, kron), transformation_matrix_conjugate)
        mueller_matrix = xp.permute_dims(mueller_matrix, (1, 2, 0))
        mueller_matrix = xp.real(mueller_matrix)

        # Restore the grid on the resulting Field.
        if hasattr(mueller_matrix, 'grid'):
            mueller_matrix.grid = jones_matrix.grid
            return mueller_matrix

        return Field(mueller_matrix, jones_matrix.grid)
    else:
        # Kronecker product of the Jones matrix with its conjugate.
        kron = data[:, None, :, None] * data_conjugate[None, :, None, :]
        kron = xp.reshape(kron, (4, 4))

        # Transform the Kronecker product into a Mueller matrix.
        mueller_matrix = xp.matmul(xp.matmul(transformation_matrix, kron), transformation_matrix_conjugate)
        mueller_matrix = xp.real(mueller_matrix)

        return mueller_matrix
