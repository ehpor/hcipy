import numpy as np

from .fourier_transform import FourierTransform, multiplex_for_tensor_fields, _get_float_and_complex_dtype
from ..field import Field
from ..config import Configuration

class NaiveFourierTransform(FourierTransform):
    '''The naive Fourier transform (NFT).

    This Fourier transform can operate on any types of coordinates, and calculates
    the Fourier integral using brute force. This class is written as generally as
    possible to act as a "ground truth" for the other Fourier transforms. As such,
    it has terrible performance.

    .. note::
        In almost all cases, you should not use this class. The FastFourierTransform and
        MatrixFourierTransform classes, if applicable, are multiple orders of magnitudes faster
        and use multiple orders of magnitude less memory. This class should only be used in a last
        resort and if you know what you are doing.

    .. note::
        The transformation matrices can be very large, quickly overwhelming any computer even
        for relatively small input or output grids.

    Parameters
    ----------
    input_grid : Grid
        The grid that is expected for the input field.
    output_grid : Grid
        The grid that is produced by the Fourier transform.
    precompute_matrices : boolean or None
        Whether to precompute the matrices used in the NFT. Turning this on will cache calculated matrices
        for future evaluations of the Fourier transform. This will use a large amount of memory. If this is
        None, the choice will be determined by the configuration file.

    Raises
    ------
    ValueError
        If the output grid has a different dimension than the input grid.
    '''
    def __init__(self, input_grid, output_grid, precompute_matrices=None):
        if input_grid.ndim != output_grid.ndim:
            raise ValueError('The input_grid must have the same dimensions as the output_grid.')

        self.input_grid = input_grid
        self.output_grid = output_grid

        self._matrix_forward = None
        self._matrix_backward = None

        if precompute_matrices is None:
            precompute_matrices = Configuration().fourier.nft.precompute_matrices
        self.precompute_matrices = precompute_matrices

        self.coords_in = np.array(self.input_grid.as_('cartesian').coords)
        self.coords_out = np.array(self.output_grid.as_('cartesian').coords)

    @property
    def matrix_forward(self):
        '''The cached forward propagation matrix.
        '''
        if not self.precompute_matrices:
            return self.get_transformation_matrix_forward()

        if self._matrix_forward is None:
            self._matrix_forward = self.get_transformation_matrix_forward()

        return self._matrix_forward

    @property
    def matrix_backward(self):
        '''The cached backward propagation matrix.
        '''
        if not self.precompute_matrices:
            return self.get_transformation_matrix_backward()

        if self._matrix_backward is None:
            self._matrix_backward = self.get_transformation_matrix_backward()

        return self._matrix_backward

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
        if self.precompute_matrices:
            res = self.matrix_forward.dot(field.ravel())
        else:
            res = np.array([(field * self.input_grid.weights).dot(np.exp(-1j * np.dot(p, self.coords_in))) for p in self.coords_out.T])

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
        --------
        Field
            The inverse Fourier transform of the field.
        '''
        if self.precompute_matrices:
            res = self.matrix_backward.dot(field.ravel())
        else:
            res = np.array([(field * self.output_grid.weights).dot(np.exp(1j * np.dot(p, self.coords_out))) for p in self.coords_in.T])
            res /= (2 * np.pi)**self.input_grid.ndim

        float_dtype, complex_dtype = _get_float_and_complex_dtype(field.dtype)
        return Field(res, self.input_grid).astype(complex_dtype, copy=False)
