import numpy as np
from scipy.linalg import blas
from .fourier_transform import FourierTransform, ComputationalComplexity, multiplex_for_tensor_fields, _get_float_and_complex_dtype
from ..field import Field, NewStyleField
from ..config import Configuration
from .._math.backends import array_namespace
from .._math.fourier import dft_matrix_regular, dft_matrix_separated
from array_api_compat import device


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
        determined by the configuration file.
    allocate_intermediate : boolean or None
        Whether to reserve memory for the intermediate result for the MFT. This provides a 5-10%
        speedup in exchange for higher memory usage. If this is None, the choice will be determined
        by the configuration file. This option applies to the legacy NumPy Field path. Array API
        backends use portable matrix multiplication, whose intermediate allocation is controlled
        by the backend.

    Notes
    -----
    Explicit scalar :class:`~hcipy.field.NewStyleField` inputs keep matrix products on their
    array backend. Grid geometry and matrix construction remain CPU setup operations; matrices
    and quadrature weights are uploaded and cached by namespace, device and precision when
    `precompute_matrices` is enabled. The legacy :class:`~hcipy.field.Field` path retains its
    existing SciPy BLAS implementation. Backend tensor batching is deferred to the native
    Fourier batching work.

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
        self._array_api_key = None
        self._array_api_M = None
        self._array_api_M1 = None
        self._array_api_M2 = None
        self._array_api_weights_input = None
        self._array_api_weights_output = None
        self._remove_matrices()

    def _compute_matrices(self, dtype, allocate_intermediate=True):
        '''Compute the matrices for the MFT using the specified data type.

        Parameters
        ---
        dtype : numpy data type
            The data type for which to calculate the matrices.
        allocate_intermediate : boolean
            Whether to allocate the legacy NumPy BLAS intermediate. Array API
            operations leave intermediate allocation to the backend matmul.
        '''
        # Set the correct complex and real data type, based on the input data type.
        float_dtype, complex_dtype = _get_float_and_complex_dtype(dtype)

        # Check if the matrices need to be (re)calculated.
        if self.matrices_dtype != complex_dtype:
            self.weights_input = (self.input_grid.weights).astype(float_dtype, copy=False)
            self.weights_output = (self.output_grid.weights / (2 * np.pi) ** self.ndim).astype(float_dtype, copy=False)

            # If all input weights are all the same, use a scalar instead.
            if not np.isscalar(self.weights_input) and np.all(self.weights_input == self.weights_input[0]):
                self.weights_input = self.weights_input[0]

            # If all output weights are all the same, use a scalar instead.
            if not np.isscalar(self.weights_output) and np.all(self.weights_output == self.weights_output[0]):
                self.weights_output = self.weights_output[0]

            if self.ndim == 1:
                if self.input_grid.is_regular and self.output_grid.is_regular:
                    self.M = dft_matrix_regular(
                        self.output_grid.zero[0],
                        self.input_grid.zero[0],
                        self.output_grid.delta[0],
                        self.input_grid.delta[0],
                        self.output_grid.size,
                        self.input_grid.size,
                        np,
                        np.dtype(complex_dtype),
                        conjugate=True,
                    )
                else:
                    self.M = dft_matrix_separated(self.output_grid.x, self.input_grid.x, conjugate=True).astype(complex_dtype, copy=False)
            elif self.ndim == 2:
                if self.input_grid.is_regular and self.output_grid.is_regular:
                    delta_in, dims_in, zero_in = self.input_grid.regular_coords
                    delta_out, dims_out, zero_out = self.output_grid.regular_coords

                    self.M1 = dft_matrix_regular(
                        zero_out[1], zero_in[1], delta_out[1], delta_in[1], dims_out[1], dims_in[1], np, np.dtype(complex_dtype), conjugate=True
                    )
                    self.M2 = dft_matrix_regular(
                        zero_in[0], zero_out[0], delta_in[0], delta_out[0], dims_in[0], dims_out[0], np, np.dtype(complex_dtype), conjugate=True
                    )
                else:
                    x, y = self.input_grid.coords.separated_coords
                    u, v = self.output_grid.coords.separated_coords

                    self.M1 = dft_matrix_separated(v, y, conjugate=True).astype(complex_dtype, copy=False)
                    self.M2 = dft_matrix_separated(x, u, conjugate=True).astype(complex_dtype, copy=False)

            self.matrices_dtype = complex_dtype

        # Check if the intermediate array needs to be (re)allocated.
        if allocate_intermediate and self.intermediate_dtype != complex_dtype:
            if self.ndim == 2:
                self.intermediate_array = np.empty((self.input_grid.shape[0], self.M2.shape[1]), dtype=complex_dtype)

                self.intermediate_dtype = complex_dtype

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
            self._array_api_key = None
            self._array_api_M = None
            self._array_api_M1 = None
            self._array_api_M2 = None
            self._array_api_weights_input = None
            self._array_api_weights_output = None

        if not self.allocate_intermediate:
            if self.ndim == 2:
                self.intermediate_array = None
                self.intermediate_dtype = None

    def _compute_array_api_matrices(self, data):
        '''Upload and cache matrices and weights for an Array API field.

        Matrix geometry remains a CPU setup boundary. Cached arrays are keyed by
        namespace name, device and precision so that an MFT can safely alternate
        between supported backends and dtypes.
        '''
        xp = array_namespace(data)
        complex_dtype = xp.complex64 if data.dtype in (xp.float32, xp.complex64) else xp.complex128
        key = (xp.__name__, device(data), complex_dtype)
        if self._array_api_key == key:
            return xp, complex_dtype

        target_device = device(data)
        self._array_api_weights_input = xp.asarray(
            self.weights_input, dtype=xp.float32 if complex_dtype == xp.complex64 else xp.float64, device=target_device
        )
        self._array_api_weights_output = xp.asarray(
            self.weights_output, dtype=xp.float32 if complex_dtype == xp.complex64 else xp.float64, device=target_device
        )
        if self.ndim == 1:
            self._array_api_M = xp.asarray(self.M, dtype=complex_dtype, device=target_device)
            self._array_api_M1 = None
            self._array_api_M2 = None
        else:
            self._array_api_M = None
            self._array_api_M1 = xp.asarray(self.M1, dtype=complex_dtype, device=target_device)
            self._array_api_M2 = xp.asarray(self.M2, dtype=complex_dtype, device=target_device)
        self._array_api_key = key
        return xp, complex_dtype

    def _operation_array_api(self, field, inverse):
        '''Transform a backend field using cached resident matrices.'''
        data = field.data
        xp = array_namespace(data)
        cpu_dtype = np.dtype('complex64') if data.dtype in (xp.float32, xp.complex64) else np.dtype('complex128')
        self._compute_matrices(cpu_dtype, allocate_intermediate=False)
        xp, complex_dtype = self._compute_array_api_matrices(data)
        data = xp.astype(data, complex_dtype, copy=False)
        weights = self._array_api_weights_output if inverse else self._array_api_weights_input
        data = data * weights

        if self.ndim == 1:
            matrix = self._array_api_M
            if inverse:
                result = xp.conj(xp.matmul(xp.permute_dims(matrix, (1, 0)), xp.conj(data)))
            else:
                result = xp.matmul(matrix, data)
        else:
            data = xp.reshape(data, self.shape_output if inverse else self.shape_input)
            M1 = self._array_api_M1
            M2 = self._array_api_M2
            if inverse:
                M1 = xp.permute_dims(M1, (1, 0))
                M2 = xp.permute_dims(M2, (1, 0))
                result = xp.conj(xp.matmul(M1, xp.matmul(xp.conj(data), M2)))
            else:
                result = xp.matmul(M1, xp.matmul(data, M2))
            result = xp.reshape(result, (-1,))

        self._remove_matrices()
        return NewStyleField(result, self.input_grid if inverse else self.output_grid)

    @multiplex_for_tensor_fields
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
        if isinstance(field, NewStyleField):
            return self._operation_array_api(field, inverse=False)

        self._compute_matrices(field.dtype)
        field = field.astype(self.matrices_dtype, copy=False)

        if self.ndim == 1:
            f = field * self.weights_input
            res = np.dot(self.M, f)
        elif self.ndim == 2:
            # Use handcoded BLAS call. BLAS is better when all inputs are Fortran ordered,
            # so we apply matrix multiplications on the transpose of each of the arrays
            # (which are C ordered).
            if field.dtype == 'complex64':
                gemm = blas.cgemm
            else:
                gemm = blas.zgemm

            if np.isscalar(self.weights_input):
                # Weights can be included in the gemm call as that multiplication
                # happens anyway (and it saves an array copy).
                f = field.reshape(self.shape_input)
                alpha = self.weights_input
            else:
                # Fallback in case the weights is not a scalar.
                f = (field * self.weights_input).reshape(self.shape_input)
                alpha = 1

            gemm(1, self.M2.T, f.T, c=self.intermediate_array.T, overwrite_c=True)
            res = gemm(alpha, self.intermediate_array.T, self.M1.T).T.reshape(-1)

        self._remove_matrices()

        return Field(res, self.output_grid)

    @multiplex_for_tensor_fields
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
        if isinstance(field, NewStyleField):
            return self._operation_array_api(field, inverse=True)

        self._compute_matrices(field.dtype)
        field = field.astype(self.matrices_dtype, copy=False)

        if self.ndim == 1:
            f = field * self.weights_output
            res = np.dot(self.M.conj().T, f)
        elif self.ndim == 2:
            # Use handcoded BLAS call. BLAS is better when all inputs are Fortran ordered,
            # so we apply matrix multiplications on the transpose of each of the arrays
            # (which are C ordered). Adjoint is handled by GEMM, which avoids an array
            # copy for these array as well.
            if field.dtype == 'complex64':
                gemm = blas.cgemm
            else:
                gemm = blas.zgemm

            if np.isscalar(self.weights_output):
                # Weights can be included in the gemm call as that multiplication
                # happens anyway (and it saves an array copy).
                f = field.reshape(self.shape_output)
                alpha = self.weights_output
            else:
                # Fallback in case the weights is not a scalar.
                f = (field * self.weights_output).reshape(self.shape_output)
                alpha = 1

            # Use trans_a=2 and trans_b=2 to apply the conjugte transpose on the a and b arrays.
            gemm(1, f.T, self.M1.T, trans_b=2, c=self.intermediate_array.T, overwrite_c=True)
            res = gemm(alpha, self.M2.T, self.intermediate_array.T, trans_a=2).T.reshape(-1)

        self._remove_matrices()

        return Field(res, self.input_grid)

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
            num_multiplications=num_multiplications, num_additions=num_additions, expected_execution_time=expected_execution_time
        )
