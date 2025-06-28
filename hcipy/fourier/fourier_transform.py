import numpy as np
from ..field import Field

from dataclasses import dataclass

@dataclass
class ComputationalComplexity:
    num_multiplications: int = 0
    num_additions: int = 0

    @property
    def num_operations(self):
        return self.num_multiplications + self.num_additions

class FourierTransform(object):
    '''The base class for all Fourier transform implementations.

    Attributes
    ----------
    input_grid : Grid
        The grid that is expected for the input field.
    output_grid : Grid
        The grid that is produced by the Fourier transform.
    '''
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
        raise NotImplementedError()

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
        raise NotImplementedError()

    def get_transformation_matrix_forward(self):
        '''Returns the transformation matrix corresonding to the
        Fourier transform.

        Returns
        -------
        ndarray
            A matrix representing the Fourier transform.
        '''
        coords_in = self.input_grid.as_('cartesian').coords
        coords_out = self.output_grid.as_('cartesian').coords

        A = np.exp(-1j * np.dot(np.array(coords_out).T, coords_in))
        A *= self.input_grid.weights

        return A

    def get_transformation_matrix_backward(self):
        '''Returns the transformation matrix corresonding to the
        Fourier transform.

        Returns
        -------
        ndarray
            A matrix representing the Fourier transform.
        '''
        coords_in = self.input_grid.as_('cartesian').coords
        coords_out = self.output_grid.as_('cartesian').coords

        A = np.exp(1j * np.dot(np.array(coords_in).T, coords_out))
        A *= self.output_grid.weights
        A /= (2 * np.pi)**self.input_grid.ndim

        return A

    @classmethod
    def is_supported(cls, input_grid, output_grid):
        '''Are the specified grids supported by this Fourier transform?

        Parameters
        ----------
        input_grid : Grid
            The grid that is expected for the input field.
        output_grid : Grid
            The grid that is produced by the Fourier transform.

        Returns
        -------
        boolean
            Whether the grids are supported or not.
        '''
        try:
            cls.check_if_supported(input_grid, output_grid)
        except ValueError:
            return False

        return True

    @classmethod
    def check_if_supported(cls, input_grid, output_grid):
        '''Check if the specified grids are supported by this Fourier transform.

        Parameters
        ----------
        input_grid : Grid
            The grid that is expected for the input field.
        output_grid : Grid
            The grid that is produced by the Fourier transform.

        Raises
        ------
        ValueError
            If the grids are not supported. The message will indicate why
            the grids are not supported.
        '''
        raise NotImplementedError()

    @classmethod
    def compute_complexity(cls, input_grid, output_grid):
        '''Compute the algorithmic complexity for this Fourier transform.

        Parameters
        ----------
        input_grid : Grid
            The grid that is expected for the input field.
        output_grid : Grid
            The grid that is produced by the Fourier transform.

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
        raise NotImplementedError()

def _time_it_iterative(function, num_iterations):
    import time

    start = time.perf_counter()
    for _ in range(num_iterations):
        function()
    end = time.perf_counter()

    return (end - start) / num_iterations

def _time_it(function, t_max=0.1, repeat_max=5):
    num_iterations = 1

    while True:
        time_per_iteration = _time_it_iterative(function, num_iterations)

        if time_per_iteration * num_iterations > t_max:
            break
        else:
            num_iterations *= 2

    # Shortcut if one iteration of the function itself takes longer than repeat_max * t_max.
    if num_iterations == 1 and time_per_iteration > (repeat_max * t_max):
        return time_per_iteration

    times = [time_per_iteration]

    for _ in range(repeat_max - 1):
        times.append(_time_it_iterative(function, num_iterations))

    return np.median(times)

def make_fourier_transform(input_grid, output_grid=None, q=1, fov=1, shift=0, planner='estimate'):
    '''Construct a FourierTransform object.

    The most time-efficient Fourier transform method will be chosen according to actual or estimated performance.

    Parameters
    ----------
    input_grid : Grid
        The grid that will be used for the Field passed to the Fourier transform.
    output_grid : None or Grid
        The grid of the resulting field. If it is None, a optimal grid will be chosen, according to `q` and `fov`.
    q : scalar or ndarray
        Describes how many samples to take in the Fourier domain. A value of 1 means critcally sampled in the Fourier domain.
    fov : scalar or ndarray
        Describes how far out the Fourier domain extends. A value of 1 means the same amount of samples as the spatial domain.
    shift : scalar or ndarray
        Describes by how much the Fourier domain should be shifted compared to the native sampling of FFT.
    planner : string
        If it is 'estimate', performance of the different methods will be estimated from theoretical complexity estimates.
        If it is 'measure', actual Fourier transforms will be performed to get the actual performance. The latter takes longer,
        but is more accurate.

    Returns
    -------
    FourierTransform
        The Fourier transform that was requested.
    '''
    from .fast_fourier_transform import FastFourierTransform, make_fft_grid, get_fft_parameters
    from .matrix_fourier_transform import MatrixFourierTransform
    from .naive_fourier_transform import NaiveFourierTransform
    from .zoom_fast_fourier_transform import ZoomFastFourierTransform

    options = {
        'fft': FastFourierTransform,
        'mft': MatrixFourierTransform,
        'naive': NaiveFourierTransform,
        'zft': ZoomFastFourierTransform,
    }

    if output_grid is not None:
        # Try to detect if the grid is compatible with an FFT grid.
        try:
            q, fov, shift = get_fft_parameters(output_grid, input_grid)
        except ValueError:
            # The grid is not a native FFT grid, so remove the FFT option.
            del options['fft']
    else:
        output_grid = make_fft_grid(input_grid, q, fov, shift)

    # Find the time taken for each of the options.
    expected_time = {}
    for method, ft in options.items():
        if not ft.is_supported(input_grid, output_grid):
            continue

        if planner == 'estimate':
            # Estimate the time taken based on the computational complexity.
            complexity = ft.compute_complexity(input_grid, output_grid)
            expected_time[method] = complexity.num_operations
        elif planner == 'measure':
            if method == 'naive':
                # This is very expensive to measure and should only be used if
                # it's the only option.
                expected_time[method] = np.inf
                continue

            # Construct the Fourier transform object.
            if method == 'fft':
                ft_obj = ft(input_grid, q, fov, shift)
            else:
                ft_obj = ft(input_grid, output_grid)

            # Time the Fourier transform.
            a = input_grid.zeros(dtype='complex')
            expected_time[method] = _time_it(lambda: ft_obj.forward(a))

    # Choose the best option.
    best_method = min(expected_time, key=expected_time.get)

    # Make the Fourier transform
    if best_method == 'fft':
        return FastFourierTransform(input_grid, q, fov, shift)
    else:
        return options[best_method](input_grid, output_grid)

def multiplex_for_tensor_fields(func):
    '''A decorator for automatically multiplexing a function over the tensor directions.

    This function is used internally for simplifying the implementation of the Fourier transforms.

    Parameters
    ----------
    func : function
        The function to multiplex. This function gets called for each of the tensor elements.
    '''
    def inner(self, field):
        if field.is_scalar_field:
            return func(self, field)
        else:
            f = field.reshape((-1, field.grid.size))
            res = [func(self, ff) for ff in f]
            new_shape = np.concatenate((field.tensor_shape, [-1]))
            return Field(np.array(res).reshape(new_shape), res[0].grid)

    return inner

def _get_float_and_complex_dtype(dtype):
    '''Return the floating point and complex numpy data types with the same
    bit depth per component.

    Parameters
    ----------
    dtype : numpy dtype
        The dtype to convert to a floating point and complex dtype.

    Returns
    -------
    float_dtype : numpy dtype
        The floating point dtype.
    complex_dtype : numpy dtype
        The complex dtype.
    '''
    if dtype == np.dtype('float32') or dtype == np.dtype('complex64'):
        complex_dtype = 'complex64'
        float_dtype = 'float32'
    else:
        complex_dtype = 'complex128'
        float_dtype = 'float64'

    return float_dtype, complex_dtype
