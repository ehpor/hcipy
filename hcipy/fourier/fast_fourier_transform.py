from __future__ import division
from collections.abc import Iterable

import numpy as np
from .fourier_transform import FourierTransform, ComputationalComplexity, multiplex_for_tensor_fields, _get_float_and_complex_dtype
from ..field import Field, CartesianGrid, RegularCoords
from ..config import Configuration
import numexpr as ne

from .._math import fft as _fft_module

def make_fft_grid(input_grid, q=1, fov=1, shift=0):
    '''Calculate the grid returned by a Fast Fourier Transform.

    Parameters
    ----------
    input_grid : Grid
        The grid defining the sampling in the real domain..
    q : scalar or array_like
        The amount of zeropadding to perform. A value of 1 denotes no zeropadding.
    fov : scalar or array_like
        The amount of cropping to perform in the Fourier domain.
    shift : scalar or array_like
        The shift to apply on the output grid.

    Returns
    -------
    Grid
        The grid defining the sampling in the Fourier domain.
    '''
    if not isinstance(q, Iterable):
        q = (q,) * input_grid.ndim
    if not isinstance(fov, Iterable):
        fov = (fov,) * input_grid.ndim
    if not isinstance(shift, Iterable):
        shift = (shift,) * input_grid.ndim

    # Check assumptions
    if not input_grid.is_regular:
        raise ValueError('The input_grid must be regular.')
    if not input_grid.is_('cartesian'):
        raise ValueError('The input_grid must be cartesian.')

    # Correct q for a discrete zero padding of the input grid.
    q = tuple(round(q_i * d_in) / d_in for q_i, d_in in zip(q, input_grid.dims))

    delta = tuple((2 * np.pi / (d_in * s_in)) / q_i for d_in, s_in, q_i in zip(input_grid.delta, input_grid.dims, q))
    dims = tuple(int(d_in * f_i * q_i) for d_in, f_i, q_i in zip(input_grid.dims, fov, q))
    zero = tuple(d_i * (-dim / 2 + (dim % 2) * 0.5) + s_i for d_i, dim, s_i in zip(delta, dims, shift))

    return CartesianGrid(RegularCoords(delta, dims, zero))

def get_fft_parameters(fft_grid, input_grid):
    '''Try to reconstruct the FFT parameters of a grid.

    .. note::
        Not every grid is an FFT grid. This function will raise a
        ValueError if this is the case. You can alternatively use
        `is_fft_grid()` to check if a grid is an FFT grid or not.

    .. note::
        The parameters that this function outputs might not
        correspond perfectly to the original FFT parameters you used.
        However, it guarantees that an FFT grid generated with these
        reconstructed parameters will create the same FFT grid as an
        FFT generated with the original parameters.

    Parameters
    ----------
    fft_grid : Grid
        A grid that corresponds to a native FFT grid of `input_grid`.
    input_grid : Grid
        The grid defining the sampling in the real domain.

    Returns
    -------
    q : ndarray
        The amount of zeropadding detected in the real domain.
    fov : ndarray
        The amount of cropping detected in the Fourier domain.
    shift : ndarray
        The amount of shifting detected in the Fourier domain.

    Raises
    ------
    ValueError
        If `fft_grid` does not correspond to an FFT grid of `input_grid`.
    '''
    if not input_grid.is_regular:
        raise ValueError('The input grid must be regular to reconstruct an fft grid.')
    if not fft_grid.is_regular:
        raise ValueError('The fft grid is not regular and therefore cannot be an fft grid.')
    if input_grid.ndim != fft_grid.ndim:
        raise ValueError('The fft grid does not have the same number of dimensions as input_grid.')

    q = tuple((2 * np.pi / (d_in * s_in)) / f_delta for d_in, s_in, f_delta in zip(input_grid.delta, input_grid.dims, fft_grid.delta))

    if any(q_i < 1 for q_i in q):
        raise ValueError(f'fft_grid is not an FFT grid of input_grid: q of {q} would be < 1.')

    # Check that the calculated q corresponds to an integer zeropadding.
    zeropadded_dims = tuple(q_i * d_in for q_i, d_in in zip(q, input_grid.dims))
    if any(abs(zn - round(zn)) > 1e-10 for zn in zeropadded_dims):
        raise ValueError(f'fft_grid is not an FFT grid of input_grid: q of {q} does not correspond to an integer zeropadding.')

    # Compute fov.
    fov = tuple(f_dim / zp_dim for f_dim, zp_dim in zip(fft_grid.dims, zeropadded_dims))

    # Check if fov would be < 1.
    if any(n > int(zn + 0.5) for n, zn in zip(fft_grid.dims, zeropadded_dims)):
        raise ValueError(f'fft_grid is not an FFT grid of input_grid: fov of {fov} would be > 1 .')

    # Correct fov for rounding errors (floating point errors would lead to a different dims).
    dummy_fft_grid = make_fft_grid(input_grid, q, fov)
    wrong_dims = tuple(n != m for n, m in zip(dummy_fft_grid.dims, fft_grid.dims))
    fov = tuple((f_dim + 0.5) / (d_in * q_i) if wrong_dim else f for wrong_dim, f, f_dim, d_in, q_i in zip(wrong_dims, fov, fft_grid.dims, input_grid.dims, q))

    shift = tuple(f_zero - f_delta * (-f_dim / 2 + (f_dim % 2) * 0.5) for f_zero, f_delta, f_dim in zip(fft_grid.zero, fft_grid.delta, fft_grid.dims))
    return q, fov, shift

def is_fft_grid(grid, input_grid):
    '''Returns whether `grid` is a native FFT grid of `input_grid` or not.

    .. note::
        The function get_fft_parameters() can be used

    Parameters
    ----------
    grid : Grid
        The grid in the Fourier domain. This grid is checked.
    input_grid : Grid
        The grid in the real domain of the FFT.

    Returns
    -------
    boolean
        Whether `grid` is a native FFT grid of `input_grid` for some q, fov and shift.
    '''
    try:
        get_fft_parameters(grid, input_grid)
    except ValueError:
        return False
    return True

def _numexpr_grid_shift(shift, grid, out=None):
    '''Fast evaluation of np.exp(1j * np.dot(shift, grid.coords)) using NumExpr.

    Parameters
    ----------
    shift : array_like
        The coordinates of the shift.
    grid : Grid
        The grid on which to calculate the shift.
    out : array_like
        An existing array where the outcome is going to be stored. This must
        have the correct shape and dtype. No checking will be performed. If
        this is None, a new array will be allocated and returned.

    Returns
    -------
    array_like
        The calculated complex shift array.
    '''
    variables = {}
    command = []
    coords = grid.coords

    for i in range(grid.ndim):
        variables[f'a{i}'] = shift[i]
        variables[f'b{i}'] = coords[i]

        command.append(f'a{i} * b{i}')

    command = 'exp(1j * (' + '+'.join(command) + '))'
    return ne.evaluate(command, local_dict=variables, out=out)

class FastFourierTransform(FourierTransform):
    '''A Fast Fourier Transform (FFT) object.

    This Fourier transform calculates FFTs with zeropadding and cropping. This
    Fourier transform requires the input grid to be regular in Cartesian coordinates. Every
    number of dimensions is allowed.

    Parameters
    ----------
    input_grid : Grid
        The grid that is expected for the input field.
    q : scalar or array_like
        The amount of zeropadding to perform. A value of 1 denotes no zeropadding. A value of
        2 indicates zeropadding to twice the dimensions of the input grid. Note: as
        zeropadding has to be done by an integer number of pixels, the q will be rounded to
        the closest possible number to satisfy this constraint.
    fov : scalar or array_like
        The amount of cropping to perform in the Fourier domain. A value of 1 indicates that
        no cropping will be performed.
    shift : array_like or scalar
        The amount by which to shift the output grid. If this is a scalar, the same shift will
        be used for all dimensions.
    emulate_fftshifts : boolean or None
        Whether to emulate FFTshifts normally used in the FFT by multiplications in the
        opposite domain. Enabling this increases performance by 3x, but degrades accuracy of
        the FFT by 10x. If this is None, the choice will be determined by the configuration
        file.

    Raises
    ------
    ValueError
        If the input grid is not regular or Cartesian.
    ValueError
        If q < 1 or fov < 0 or fov > 1, both of which are impossible for an FFT to calculate.
    '''
    def __init__(self, input_grid, q=1, fov=1, shift=0, emulate_fftshifts=None):
        # Check assumptions
        if not input_grid.is_regular:
            raise ValueError('The input_grid must be regular.')
        if not input_grid.is_('cartesian'):
            raise ValueError('The input_grid must be Cartesian.')

        q_check = (q,) if np.isscalar(q) else q
        fov_check = (fov,) if np.isscalar(fov) else fov

        if any(q_i < 1 for q_i in q_check):
            raise ValueError('The amount of zeropadding (q) must be larger than 1.')

        if any(fov_i < 0 for fov_i in fov_check):
            raise ValueError('The amount of cropping (fov) must be positive.')

        self.input_grid = input_grid

        self.shape_in = input_grid.shape
        self.weights = input_grid.weights
        self.size = input_grid.size
        self.ndim = input_grid.ndim

        # Get the value from the configuration file if left at the default.
        if emulate_fftshifts is None:
            emulate_fftshifts = Configuration().fourier.fft.emulate_fftshifts
        self.emulate_fftshifts = emulate_fftshifts

        self.output_grid = make_fft_grid(input_grid, q, fov, shift)
        self.internal_grid = make_fft_grid(input_grid, q, 1)

        if np.any(self.output_grid.dims > self.internal_grid.dims):
            raise ValueError('The amount of cropping (fov) must be smaller than 1.')

        self.shape_out = self.output_grid.shape
        self.internal_shape = self.internal_grid.shape
        self.internal_array = np.zeros(self.internal_shape, 'complex')

        # Calculate the part of the array in which to insert the input field (for zeropadding).
        if self.internal_shape == self.shape_in:
            self.cutout_input = None
        else:
            cutout_start = tuple(int(internal_dim / 2) - int(input_dim / 2) for internal_dim, input_dim in zip(self.internal_shape, self.shape_in))
            cutout_end = tuple(start + input_dim for start, input_dim in zip(cutout_start, self.shape_in))
            self.cutout_input = tuple([slice(start, end) for start, end in zip(cutout_start, cutout_end)])

        # Calculate the part of the array to extract the output field (for cropping).
        if self.internal_shape == self.shape_out:
            self.cutout_output = None
        else:
            cutout_start = tuple(int(internal_dim / 2) - int(output_dim / 2) for internal_dim, output_dim in zip(self.internal_shape, self.shape_out))
            cutout_end = tuple(start + output_dim for start, output_dim in zip(cutout_start, self.shape_out))
            self.cutout_output = tuple([slice(start, end) for start, end in zip(cutout_start, cutout_end)])

        # Calculate the shift array when the input grid was shifted compared to the native shift
        # expected by the numpy FFT implementation.
        center = tuple(zero + delta * (dim // 2) for zero, delta, dim in zip(input_grid.zero, input_grid.delta, input_grid.dims))
        self.shift_input = _numexpr_grid_shift(tuple(-c for c in center), self.output_grid)

        # Remove piston shift (remove central shift phase)
        self.shift_input /= np.fft.ifftshift(self.shift_input.reshape(self.shape_out)).ravel()[0]

        # Calculate the multiplication for emulating the FFTshift (if requested).
        if emulate_fftshifts:
            f_shift = tuple(delta * (dim // 2) for delta, dim in zip(input_grid.delta, self.internal_shape[::-1]))
            fftshift = _numexpr_grid_shift(f_shift, self.internal_grid)

            if self.cutout_output:
                self.shift_input *= fftshift.reshape(self.internal_shape)[self.cutout_output].ravel()
            else:
                self.shift_input *= fftshift

        # Apply weights for Fourier normalization.
        self.shift_input *= self.weights

        # Calculate the shift array when the output grid was shifted compared to the native shift
        # expcted by the numpy FFT implementation.
        shift = np.ones(self.input_grid.ndim) * shift
        if np.allclose(shift, 0):
            self.shift_output = 1
        else:
            self.shift_output = _numexpr_grid_shift(-shift, self.input_grid)

        # Calculate the multiplication for emulating the FFTshift (if requested).
        if emulate_fftshifts:
            f_shift = tuple(delta * (dim // 2) for delta, dim in zip(self.input_grid.delta, self.internal_shape[::-1]))
            fftshift = _numexpr_grid_shift(f_shift, self.internal_grid)

            fftshift *= np.exp(-1j * np.dot(f_shift, self.internal_grid.zero))

            if self.cutout_input:
                self.shift_output *= fftshift.reshape(self.internal_shape)[self.cutout_input].ravel()
            else:
                self.shift_output *= fftshift

        # Detect if we don't need to shift in the output plane (to avoid a multiplication in the operation).
        if np.isscalar(self.shift_output) and np.allclose(self.shift_output, 1):
            self.shift_output = None

    @multiplex_for_tensor_fields
    def forward(self, field):
        '''Returns the forward Fourier transform of the :class:`Field` field.

        Parameters
        ----------
        field : Field
            The field to Fourier transform.

        Returns
        -------
        Field
            The Fourier transform of the field.
        '''
        if self.cutout_input is None:
            self.internal_array[:] = field.reshape(self.shape_in)

            if self.shift_output is not None:
                self.internal_array *= self.shift_output.reshape(self.shape_in)
        else:
            self.internal_array[:] = 0
            self.internal_array[self.cutout_input] = field.reshape(self.shape_in)

            if self.shift_output is not None:
                self.internal_array[self.cutout_input] *= self.shift_output.reshape(self.shape_in)

        if not self.emulate_fftshifts:
            self.internal_array = np.fft.ifftshift(self.internal_array)

        fft_array = _fft_module.fftn(self.internal_array)

        if not self.emulate_fftshifts:
            fft_array = np.fft.fftshift(fft_array)

        if self.cutout_output is None:
            res = fft_array.ravel()
        else:
            res = fft_array[self.cutout_output].ravel()

        res *= self.shift_input

        float_dtype, complex_dtype = _get_float_and_complex_dtype(field.dtype)
        return Field(res, self.output_grid).astype(complex_dtype, copy=False)

    @multiplex_for_tensor_fields
    def backward(self, field):
        '''Returns the inverse Fourier transform of the :class:`Field` field.

        Parameters
        ----------
        field : Field
            The field to inverse Fourier transform.

        Returns
        -------
        Field
            The inverse Fourier transform of the field.
        '''
        if self.cutout_output is None:
            self.internal_array[:] = field.reshape(self.shape_out)
            self.internal_array /= self.shift_input.reshape(self.shape_out)
        else:
            self.internal_array[:] = 0
            self.internal_array[self.cutout_output] = field.reshape(self.shape_out)
            self.internal_array[self.cutout_output] /= self.shift_input.reshape(self.shape_out)

        if not self.emulate_fftshifts:
            self.internal_array = np.fft.ifftshift(self.internal_array)

        fft_array = _fft_module.ifftn(self.internal_array)

        if not self.emulate_fftshifts:
            fft_array = np.fft.fftshift(fft_array)

        if self.cutout_input is None:
            res = fft_array.ravel()
        else:
            res = fft_array[self.cutout_input].ravel()

        if self.shift_output is not None:
                res /= self.shift_output

        float_dtype, complex_dtype = _get_float_and_complex_dtype(field.dtype)
        return Field(res, self.input_grid).astype(complex_dtype, copy=False)

    @classmethod
    def check_if_supported(cls, input_grid, output_grid):
        '''Check if the specified grids are supported by the Fast Fourier transform.

        Parameters
        ----------
        input_grid : Grid
            The grid that is expected for the input field.
        output_grid : Grid
            The grid that is produced by the Fast Fourier transform.

        Raises
        ------
        ValueError
            If the grids are not supported. The message will indicate why
            the grids are not supported.
        '''
        get_fft_parameters(output_grid, input_grid)

    @classmethod
    def compute_complexity(cls, input_grid, output_grid):
        '''Compute the algorithmic complexity for the Fast Fourier transform.

        Parameters
        ----------
        input_grid : Grid
            The grid that is expected for the input field.
        output_grid : Grid
            The grid that is produced by the Fast Fourier transform.

        Returns
        -------
        AlgorithmicComplexity
            The algorithmic complexity for this Fourier transform.

        Raises
        ------
        ValueError
            If the grids are not supported. The message will indicate why
            the grids are not supported.
        '''
        q, _, shift = get_fft_parameters(output_grid, input_grid)

        shape = input_grid.shape.astype('float') * q

        N_internal = np.prod(shape)
        N_input = np.prod(input_grid.shape)
        N_output = np.prod(output_grid.shape)

        num_complex_multiplications = 0.5 * N_internal * np.log2(N_internal)
        num_complex_additions = N_internal * np.log2(N_internal)

        # Add complexity for initial multiplication by shift_output.
        # The multiplication happens on `field.reshape(self.shape_in)` which has N_input elements.
        if not np.allclose(shift, 0):
            num_complex_multiplications += N_input

        # Add complexity for final multiplication by shift_input
        # This multiplication happens on `res` which has N_output elements.
        num_complex_multiplications += N_output

        # Convert to real operations
        num_multiplications = 4 * num_complex_multiplications
        num_additions = 2 * num_complex_multiplications + 2 * num_complex_additions
        num_operations = num_multiplications + num_additions

        # Predict execution time.
        prediction_coefficients = Configuration().fourier.fft.execution_time_prediction_coefficients
        expected_execution_time = FourierTransform._predict_execution_time(num_operations, prediction_coefficients)

        return ComputationalComplexity(
            num_multiplications=num_multiplications,
            num_additions=num_additions,
            expected_execution_time=expected_execution_time
        )
