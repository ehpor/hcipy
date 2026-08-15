from __future__ import division
from collections.abc import Iterable

import numpy as np
from .fourier_transform import FourierTransform, ComputationalComplexity, _get_float_and_complex_dtype
from ..field import Field, CartesianGrid, RegularCoords
from ..config import Configuration
import math

from .._math.separable_filter import make_separable_filter
from .._math import fft as _fft_module


def _make_shift_filter(slopes, grid, internal_grid, cutout, f_shift, piston=0.0, scale=1):
    '''Build a separable filter for one side of the FFT.

    The filter is `exp(1j * (slope * x + const))` per axis, where `slope`
    and `const` include the FFTshift-emulation phase: the emulation phase
    `exp(1j * f_shift * x_internal)`, restricted to the cutout region of the
    internal grid, is a phase ramp on `grid` with slope
    `f_shift * delta_internal / delta` plus a constant phase. The constant
    phase is folded into the factors; the piston phase is removed and
    `scale` multiplies the first factor.
    '''
    ratio = internal_grid.delta / grid.delta
    slopes = slopes + f_shift * ratio
    starts = np.array([cutout[grid.ndim - 1 - a].start if cutout is not None else 0 for a in range(grid.ndim)])
    consts = f_shift * (internal_grid.zero + starts * internal_grid.delta - ratio * grid.zero)

    factors = [np.exp(1j * (slopes[a] * x + consts[a])) for a, x in enumerate(grid.separated_coords)]
    factors[-1] = factors[-1] * np.exp(1j * piston) * scale
    return make_separable_filter(tuple(reversed(factors)), threshold=grid.size)

def _allclose(a, b, rtol=1e-5, atol=1e-8):
    if len(a) != len(b):
        return False

    for x, y in zip(a, b):
        if abs(x - y) > (atol + rtol * abs(y)):
            return False

    return True

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

    delta = tuple(float((2 * math.pi / (d_in * s_in)) / q_i) for d_in, s_in, q_i in zip(input_grid.delta, input_grid.dims, q))
    dims = tuple(int(d_in * f_i * q_i) for d_in, f_i, q_i in zip(input_grid.dims, fov, q))
    zero = tuple(float(d_i * (-dim / 2 + (dim % 2) * 0.5) + s_i) for d_i, dim, s_i in zip(delta, dims, shift))

    return CartesianGrid(RegularCoords(delta, dims, zero, xp=input_grid.xp))

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
    q : tuple
        The amount of zeropadding detected in the real domain.
    fov : tuple
        The amount of cropping detected in the Fourier domain.
    shift : tuple
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

    q = tuple((2 * math.pi / (d_in * s_in)) / f_delta for d_in, s_in, f_delta in zip(input_grid.delta, input_grid.dims, fft_grid.delta))

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
        self.internal_array = None

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

        # Emulate the FFTshifts by a phase multiplication in the opposite domain.
        xp = input_grid.xp
        f_shift = input_grid.delta * (xp.asarray(self.internal_grid.dims) // 2) if emulate_fftshifts else 0
        weights = xp.prod(input_grid.delta)

        # Build the output-side filter, when the input grid was shifted compared
        # to the native shift expected by the numpy FFT implementation. Remove
        # the piston shift (remove central shift phase).
        center = input_grid.zero + input_grid.delta * (xp.asarray(input_grid.dims) // 2)
        origins = [x_a[x_a.shape[0] // 2] for x_a in self.output_grid.separated_coords]
        self.shift_input_filter = _make_shift_filter(
            -center, self.output_grid, self.internal_grid, self.cutout_output, f_shift,
            piston=np.dot(center, origins), scale=weights)

        # Build the input-side filter, when the output grid was shifted compared
        # to the native shift expected by the numpy FFT implementation.
        shift = np.ones(self.input_grid.ndim) * shift
        if emulate_fftshifts or not np.allclose(shift, 0):
            piston = -np.dot(f_shift, self.internal_grid.zero) if emulate_fftshifts else 0.0
            self.shift_output_filter = _make_shift_filter(
                -shift, self.input_grid, self.internal_grid, self.cutout_input, f_shift, piston=piston)
        else:
            self.shift_output_filter = None

    def _compute_internal_array(self, field):
        '''(Re)allocate the internal array for the given field if necessary.

        The internal array follows the ``tensor_shape + grid.shape`` layout,
        with the leading tensor axes of the field.
        '''
        _, complex_dtype = _get_float_and_complex_dtype(field.dtype)

        tensor_shape = tuple(field.tensor_shape)

        recompute = self.internal_array is None
        recompute = recompute or (self.internal_array.dtype != complex_dtype)
        recompute = recompute or (self.internal_array.ndim != field.grid.ndim + field.tensor_order)
        recompute = recompute or (self.internal_array.shape[:field.tensor_order] != tensor_shape)

        if recompute:
            self.internal_array = np.zeros(tensor_shape + self.internal_shape, complex_dtype)

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
        self._compute_internal_array(field)

        tensor_shape = tuple(field.tensor_shape)
        c = (slice(None),) * field.tensor_order
        axes = tuple(range(-self.ndim, 0))

        if self.cutout_input is None:
            if self.shift_output_filter is None:
                self.internal_array[:] = field.reshape(tensor_shape + self.shape_in)
            else:
                self.shift_output_filter.apply_numpy(field.reshape(tensor_shape + self.shape_in), out=self.internal_array)
        else:
            self.internal_array[:] = 0
            if self.shift_output_filter is None:
                self.internal_array[c + self.cutout_input] = field.reshape(tensor_shape + self.shape_in)
            else:
                self.shift_output_filter.apply_numpy(field.reshape(tensor_shape + self.shape_in), out=self.internal_array[c + self.cutout_input])

        if not self.emulate_fftshifts:
            self.internal_array = np.fft.ifftshift(self.internal_array, axes=axes)

        fft_array = _fft_module.fftn(self.internal_array, axes=axes)

        if not self.emulate_fftshifts:
            fft_array = np.fft.fftshift(fft_array, axes=axes)

        if self.cutout_output is None:
            res = fft_array
        else:
            res = fft_array[c + self.cutout_output]

        res = self.shift_input_filter.apply_numpy(res, out=res)

        res = res.reshape(tensor_shape + (-1,))

        float_dtype, complex_dtype = _get_float_and_complex_dtype(field.dtype)
        return Field(res, self.output_grid).astype(complex_dtype, copy=False)

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
        self._compute_internal_array(field)

        tensor_shape = tuple(field.tensor_shape)
        c = (slice(None),) * field.tensor_order
        axes = tuple(range(-self.ndim, 0))

        if self.cutout_output is None:
            self.shift_input_filter.apply_numpy(field.reshape(tensor_shape + self.shape_out), out=self.internal_array, inverse=True)
        else:
            self.internal_array[:] = 0
            self.shift_input_filter.apply_numpy(field.reshape(tensor_shape + self.shape_out), out=self.internal_array[c + self.cutout_output], inverse=True)

        if not self.emulate_fftshifts:
            self.internal_array = np.fft.ifftshift(self.internal_array, axes=axes)

        fft_array = _fft_module.ifftn(self.internal_array, axes=axes)

        if not self.emulate_fftshifts:
            fft_array = np.fft.fftshift(fft_array, axes=axes)

        if self.cutout_input is None:
            res = fft_array
        else:
            res = fft_array[c + self.cutout_input]

        if self.shift_output_filter is not None:
            res = self.shift_output_filter.apply_numpy(res, out=res, inverse=True)

        res = res.reshape(tensor_shape + (-1,))

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

        shape = tuple(n * qq for n, qq in zip(input_grid.shape, q))

        N_internal = math.prod(shape)
        N_input = math.prod(input_grid.shape)
        N_output = math.prod(output_grid.shape)

        num_complex_multiplications = 0.5 * N_internal * math.log2(N_internal)
        num_complex_additions = N_internal * math.log2(N_internal)

        # Add complexity for initial multiplication by shift_output.
        # The multiplication happens on `field.reshape(self.shape_in)` which has N_input elements.
        if not _allclose(shift, (0,) * len(shift)):
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
