import numpy as np
from scipy.linalg import blas
from .fourier_transform import FourierTransform, multiplex_for_tensor_fields, _get_float_and_complex_dtype
from ..field import Field
from ..config import Configuration
import numexpr as ne

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
        by the configuration file.

    Raises
    ------
    ValueError
        If the input grid is not separated in Cartesian coordinates, if it's not one- or two-
        dimensional, or if the output grid has a different dimension than the input grid.
    '''
    def __init__(self, input_grid, output_grid, precompute_matrices=None, allocate_intermediate=None):
        # Check input grid assumptions
        if not input_grid.is_separated or not input_grid.is_('cartesian'):
            raise ValueError('The input_grid must be separable in cartesian coordinates.')
        if not output_grid.is_separated or not output_grid.is_('cartesian'):
            raise ValueError('The output_grid must be separable in cartesian coordinates.')
        if input_grid.ndim not in [1, 2]:
            raise ValueError('The input_grid must be one- or two-dimensional.')
        if input_grid.ndim != output_grid.ndim:
            raise ValueError('The input_grid must have the same dimensions as the output_grid.')

        self.input_grid = input_grid
        self.output_grid = output_grid

        self.shape_input = input_grid.shape
        self.shape_output = output_grid.shape

        self.ndim = input_grid.ndim

        # Get the value from the configuration file if left at default.
        if precompute_matrices is None:
            precompute_matrices = Configuration().fourier.mft.precompute_matrices
        self.precompute_matrices = precompute_matrices

        # Get the value from the configuration file if left at default.
        if allocate_intermediate is None:
            allocate_intermediate = Configuration().fourier.mft.allocate_intermediate
        self.allocate_intermediate = allocate_intermediate

        self.matrices_dtype = None
        self.intermediate_dtype = None
        self._remove_matrices()

    def _compute_matrices(self, dtype):
        '''Compute the matrices for the MFT using the specified data type.

        Parameters
        ---
        dtype : numpy data type
            The data type for which to calculate the matrices.
        '''
        # Set the correct complex and real data type, based on the input data type.
        float_dtype, complex_dtype = _get_float_and_complex_dtype(dtype)

        # Check if the matrices need to be (re)calculated.
        if self.matrices_dtype != complex_dtype:
            self.weights_input = (self.input_grid.weights).astype(float_dtype, copy=False)
            self.weights_output = (self.output_grid.weights / (2 * np.pi)**self.ndim).astype(float_dtype, copy=False)

            # If all input weights are all the same, use a scalar instead.
            if not np.isscalar(self.weights_input) and np.all(self.weights_input == self.weights_input[0]):
                self.weights_input = self.weights_input[0]

            # If all output weights are all the same, use a scalar instead.
            if not np.isscalar(self.weights_output) and np.all(self.weights_output == self.weights_output[0]):
                self.weights_output = self.weights_output[0]

            if self.ndim == 1:
                xu = np.outer(self.output_grid.x, self.input_grid.x)
                self.M = ne.evaluate('exp(-1j * xu)', local_dict={'xu': xu}).astype(complex_dtype, copy=False)
            elif self.ndim == 2:
                x, y = self.input_grid.coords.separated_coords
                u, v = self.output_grid.coords.separated_coords

                vy = np.outer(v, y)
                xu = np.outer(x, u)

                self.M1 = ne.evaluate('exp(-1j * vy)', local_dict={'vy': vy}).astype(complex_dtype, copy=False)
                self.M2 = ne.evaluate('exp(-1j * xu)', local_dict={'xu': xu}).astype(complex_dtype, copy=False)

            self.matrices_dtype = complex_dtype

        # Checki if the intermediate array needs to be (re)allocated.
        if self.intermediate_dtype != complex_dtype:
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

        if not self.allocate_intermediate:
            if self.ndim == 2:
                self.intermediate_array = None
                self.intermediate_dtype = None

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
