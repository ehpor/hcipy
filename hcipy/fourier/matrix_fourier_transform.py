import numpy as np
from array_api_compat import device
from .fourier_transform import FourierTransform, ComputationalComplexity
from ..field import NewStyleField
from ..config import Configuration
from .._math.backends import array_namespace, device_context
from .._math.fourier import dft_matrix_regular, dft_matrix_separated
from .._math.linalg import matmul

class MatrixFourierTransform(FourierTransform):
    '''A Matrix Fourier Transform (MFT) object.

    This Fourier transform is based on the MFT described in [Soummer2007]_. It requires both
    the input and output grid to be separated in Cartesian coordinates. Additionally, due to
    current implementation limitations, this Fourier transform only supports one- and two-dimensional
    grids.

    .. [Soummer2007] Soummer et al. 2007, "Fast computation of Lyot-style
        coronagraph propagation".

    Parameters
    ----------
    input_grid : Grid
        The grid that is expected for the input field.
    output_grid : Grid
        The grid that is produced by the Fourier transform.
    precompute_matrices : boolean or None
        Whether to precompute the matrices used in the MFT. Turning this on will provide a 20-30%
        speedup, in exchange for higher memory usage.If this is False, the matrices will be
        calculated each time a Fourier transform is performed. If this is True, the matrices will be
        calculated once, and reused for future evaluations. If this is None, the choice will be
        determined by the configuration file. Disable precomputation when grid coordinates
        change or when a fresh differentiation graph is needed for each evaluation.
    allocate_intermediate : boolean or None
        Whether to reserve memory for the intermediate result for the MFT. This provides a 5-10%
        speedup in exchange for higher memory usage. If this is None, the choice will be determined
        by the configuration file. Buffer reuse depends on backend support.

    Raises
    ------
    ValueError
        If the input grid is not separated in Cartesian coordinates, if it's not one- or two-
        dimensional, or if the output grid has a different dimension than the input grid.
    '''
    def __init__(self, input_grid, output_grid, precompute_matrices=None, allocate_intermediate=None):
        self.check_if_supported(input_grid, output_grid)

        self.input_grid = input_grid
        self.output_grid = output_grid

        self.shape_input = input_grid.shape
        self.shape_output = output_grid.shape

        self.ndim = input_grid.ndim

        # Get the value from the configuration file if left at default.
        if precompute_matrices is None:
            precompute_matrices = Configuration().mft_precompute_matrices
        self.precompute_matrices = precompute_matrices

        # Get the value from the configuration file if left at default.
        if allocate_intermediate is None:
            allocate_intermediate = Configuration().mft_allocate_intermediate
        self.allocate_intermediate = allocate_intermediate

        self.matrices_dtype = None
        self.intermediate_dtype = None
        self._matrix_key = None
        self._intermediate_key = None
        self.intermediate_array = None
        self._remove_matrices()

    def _compute_matrices(self, data):
        '''Compute the matrices on the input array's backend and device.

        Parameters
        ---
        data : Array
            The input array, defining the namespace, device and precision.
        '''
        # Set the correct complex and real data type, based on the input data type.
        xp = array_namespace(data)
        single_precision = data.dtype in (xp.float32, xp.complex64)
        float_dtype = xp.float32 if single_precision else xp.float64
        complex_dtype = xp.complex64 if single_precision else xp.complex128
        target_device = device(data)
        key = (xp.__name__, target_device, complex_dtype)

        # Check if the matrices need to be (re)calculated.
        if self._matrix_key != key:
            self.weights_input = xp.asarray(self.input_grid.weights, dtype=float_dtype, device=target_device)
            self.weights_output = xp.asarray(self.output_grid.weights, dtype=float_dtype, device=target_device) / (2 * np.pi)**self.ndim

            if self.ndim == 1:
                if self.input_grid.is_regular and self.output_grid.is_regular:
                    self.M = dft_matrix_regular(
                        self.output_grid.zero[0], self.input_grid.zero[0],
                        self.output_grid.delta[0], self.input_grid.delta[0],
                        self.shape_output[0], self.shape_input[0],
                        xp, complex_dtype, conjugate=True, device=target_device)
                else:
                    x = xp.asarray(self.input_grid.coords.separated_coords[0], device=target_device)
                    u = xp.asarray(self.output_grid.coords.separated_coords[0], device=target_device)
                    self.M = xp.astype(dft_matrix_separated(u, x, conjugate=True), complex_dtype, copy=False)
            elif self.ndim == 2:
                if self.input_grid.is_regular and self.output_grid.is_regular:
                    delta_in, dims_in, zero_in = self.input_grid.regular_coords
                    delta_out, dims_out, zero_out = self.output_grid.regular_coords

                    self.M1 = dft_matrix_regular(
                        zero_out[1], zero_in[1], delta_out[1], delta_in[1],
                        dims_out[1], dims_in[1], xp, complex_dtype, conjugate=True, device=target_device)
                    self.M2 = dft_matrix_regular(
                        zero_in[0], zero_out[0], delta_in[0], delta_out[0],
                        dims_in[0], dims_out[0], xp, complex_dtype, conjugate=True, device=target_device)
                else:
                    x, y = [xp.asarray(c, device=target_device) for c in self.input_grid.coords.separated_coords]
                    u, v = [xp.asarray(c, device=target_device) for c in self.output_grid.coords.separated_coords]

                    self.M1 = xp.astype(dft_matrix_separated(v, y, conjugate=True), complex_dtype, copy=False)
                    self.M2 = xp.astype(dft_matrix_separated(x, u, conjugate=True), complex_dtype, copy=False)

            self.matrices_dtype = complex_dtype
            self._matrix_key = key

        if self._intermediate_key != key:
            self.intermediate_array = None
            self.intermediate_dtype = None
            self._intermediate_key = key

    def _remove_matrices(self):
        '''Remove the matrices after a Fourier transform.

        This is is used to clean up the used matrices after a Fourier transform operation.
        '''
        if not self.precompute_matrices:
            if self.ndim == 1:
                self.M = None
            elif self.ndim == 2:
                self.M1 = None
                self.M2 = None

            self.matrices_dtype = None
            self._matrix_key = None

        if not self.allocate_intermediate:
            if self.ndim == 2:
                self.intermediate_array = None
                self.intermediate_dtype = None
                self._intermediate_key = None

    def forward(self, field):
        '''Returns the forward Fourier transform of the :class:`Field` field.

        Parameters
        ----------
        field : Field
            The field to Fourier transform.

        Returns
        --------
        Field
            The Fourier transform of the field.
        '''
        data = field.data if isinstance(field, NewStyleField) else field
        with device_context(data):
            xp = array_namespace(data)
            self._compute_matrices(data)
            data = xp.astype(xp.asarray(data), self.matrices_dtype, copy=False)

            if self.weights_input.ndim == 0:
                alpha = self.weights_input
            else:
                alpha = 1
                data = data * self.weights_input

            if self.ndim == 1:
                res = matmul(self.M, data[..., None], alpha=alpha)
            elif self.ndim == 2:
                # The leading tensor dimensions act as the batch. Reshape as a 2D array.
                f = xp.reshape(data, tuple(field.tensor_shape) + self.shape_input)
                self.intermediate_array = matmul(f, self.M2, out=self.intermediate_array)
                self.intermediate_dtype = self.matrices_dtype
                res = matmul(self.M1, self.intermediate_array, alpha=alpha)

            self._remove_matrices()

            return type(field)(xp.reshape(res, tuple(field.tensor_shape) + (-1,)), self.output_grid)

    def backward(self, field):
        '''Returns the inverse Fourier transform of the :class:`Field` field.

        Parameters
        ----------
        field : Field
            The field to inverse Fourier transform.

        Returns
        --------
        Field
            The inverse Fourier transform of the field.
        '''
        data = field.data if isinstance(field, NewStyleField) else field
        with device_context(data):
            xp = array_namespace(data)
            self._compute_matrices(data)
            data = xp.astype(xp.asarray(data), self.matrices_dtype, copy=False)

            if self.weights_output.ndim == 0:
                alpha = self.weights_output
            else:
                alpha = 1
                data = data * self.weights_output

            if self.ndim == 1:
                res = matmul(self.M, data[..., None], alpha=alpha, adjoint_a=True)
            elif self.ndim == 2:
                # The leading tensor dimensions act as the batch. Reshape as a 2D array.
                f = xp.reshape(data, tuple(field.tensor_shape) + self.shape_output)
                self.intermediate_array = matmul(self.M1, f, adjoint_a=True, out=self.intermediate_array)
                self.intermediate_dtype = self.matrices_dtype
                res = matmul(self.intermediate_array, self.M2, alpha=alpha, adjoint_b=True)

            self._remove_matrices()

            return type(field)(xp.reshape(res, tuple(field.tensor_shape) + (-1,)), self.input_grid)

    @classmethod
    def check_if_supported(cls, input_grid, output_grid):
        '''Check if the specified grids are supported by the Matrix Fourier transform.

        Parameters
        ----------
        input_grid : Grid
            The grid that is expected for the input field.
        output_grid : Grid
            The grid that is produced by the Matrix Fourier transform.

        Raises
        ------
        ValueError
            If the grids are not supported. The message will indicate why
            the grids are not supported.
        '''
        if not input_grid.is_separated or not input_grid.is_('cartesian'):
            raise ValueError('The input_grid must be separable in cartesian coordinates.')

        if not output_grid.is_separated or not output_grid.is_('cartesian'):
            raise ValueError('The output_grid must be separable in cartesian coordinates.')

        if input_grid.ndim not in [1, 2]:
            raise ValueError('The input_grid must be one- or two-dimensional.')

        if input_grid.ndim != output_grid.ndim:
            raise ValueError('The input_grid must have the same dimensions as the output_grid.')

    @classmethod
    def compute_complexity(cls, input_grid, output_grid):
        '''Compute the algorithmic complexity for the Matrix Fourier transform.

        Parameters
        ----------
        input_grid : Grid
            The grid that is expected for the input field.
        output_grid : Grid
            The grid that is produced by the Matrix Fourier transform.

        Returns
        -------
        AlgorithmicComplexity
            The algorithmic complexity for the Fourier transform.

        Raises
        ------
        ValueError
            If the grids are not supported. The message will indicate why
            the grids are not supported.
        '''
        cls.check_if_supported(input_grid, output_grid)

        if input_grid.ndim == 1:
            num_complex_multiplications = input_grid.size * output_grid.size
            num_complex_additions = (input_grid.size - 1) * output_grid.size
        elif input_grid.ndim == 2:
            N_out_x, N_out_y = output_grid.shape
            N_in_x, N_in_y = input_grid.shape

            # Complexity for gemm(1, self.M2.T, f.T)
            num_complex_multiplications = N_in_y * N_in_x * N_out_y
            num_complex_additions = N_out_y * N_in_x * (N_in_y - 1)

            # Complexity for gemm(alpha, self.intermediate_array.T, self.M1.T)
            num_complex_multiplications += N_in_x * N_out_x * N_out_y
            num_complex_additions += N_out_x * N_out_y * (N_in_x - 1)

            # Add complexity for initial multiplication by weights_input if not scalar
            num_complex_multiplications += input_grid.size

        # Convert to real operations.
        num_multiplications = 4 * num_complex_multiplications
        num_additions = 2 * num_complex_multiplications + 2 * num_complex_additions
        num_operations = num_multiplications + num_additions

        # Predict execution time.
        prediction_coefficients = Configuration().mft_runtime_coeffs
        expected_execution_time = FourierTransform._predict_execution_time(num_operations, prediction_coefficients)

        return ComputationalComplexity(
            num_multiplications=num_multiplications,
            num_additions=num_additions,
            expected_execution_time=expected_execution_time
        )
