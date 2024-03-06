import numpy as np

from .fast_fourier_transform import FastFourierTransform
from ..field import Field, field_dot, field_conjugate_transpose
from .._math import fft as _fft_module


class FourierFilter(object):
    '''A filter in the Fourier domain.

    The filtering is performed by Fast Fourier Transforms, but is quicker than
    the equivalent multiplication in the Fourier domain using the FastFourierTransform
    classes. It does this by avoiding redundant field multiplications that limit performance.

    Parameters
    ----------
    input_grid : Grid
        The grid that is expected for the input field.
    transfer_function : Field generator or Field
        The transfer function to use for the filter. If this is a Field, the user is responsible
        for checking that its grid corresponds to the internal grid used by this filter. The Grid
        is not checked.
    q : scalar
        The amount of zeropadding to perform in the real domain. A value
        of 1 denotes no zeropadding. Zeropadding increases the resolution in the
        Fourier domain and therefore reduces aliasing/wrapping effects.
    '''
    def __init__(self, input_grid, transfer_function, q=1):
        fft = FastFourierTransform(input_grid, q)

        self.input_grid = input_grid
        self.internal_grid = fft.output_grid
        self.cutout = fft.cutout_input
        self.shape_in = input_grid.shape

        self.transfer_function = transfer_function

        self._transfer_function = None
        self.internal_array = None

    def _compute_functions(self, field):
        if self._transfer_function is None or self._transfer_function.dtype != field.dtype:
            if hasattr(self.transfer_function, '__call__'):
                tf = self.transfer_function(self.internal_grid)
            else:
                tf = self.transfer_function.copy()

            tf = np.fft.ifftshift(tf.shaped, axes=tuple(range(-self.input_grid.ndim, 0)))
            self._transfer_function = tf.astype(field.dtype, copy=False)

        recompute_internal_array = self.internal_array is None
        recompute_internal_array = recompute_internal_array or (self.internal_array.ndim != (field.grid.ndim + field.tensor_order))
        recompute_internal_array = recompute_internal_array or (self.internal_array.dtype != field.dtype)
        recompute_internal_array = recompute_internal_array or not np.array_equal(self.internal_array.shape[:field.tensor_order], field.tensor_shape)

        if recompute_internal_array:
            self.internal_array = self.internal_grid.zeros(field.tensor_shape, field.dtype).shaped

    def forward(self, field):
        '''Return the forward filtering of the input field.

        Parameters
        ----------
        field : Field
            The field to filter.

        Returns
        -------
        Field
            The filtered field.
        '''
        return self._operation(field, adjoint=False)

    def backward(self, field):
        '''Return the backward (adjoint) filtering of the input field.

        Parameters
        ----------
        field : Field
            The field to filter.

        Returns
        -------
        Field
            The adjoint filtered field.
        '''
        return self._operation(field, adjoint=True)

    def _operation(self, field, adjoint):
        '''The internal filtering operation.

        Parameters
        ----------
        field : Field
            The input field.
        adjoint : boolean
            Whether to perform a forward or adjoint filter.

        Returns
        -------
        Field
            The filtered field.
        '''
        self._compute_functions(field)

        if self.cutout is None:
            f = field.shaped
        else:
            f = self.internal_array
            f[:] = 0
            c = tuple([slice(None)] * field.tensor_order) + self.cutout
            f[c] = field.shaped

        # Don't overwrite f if it shares memory with the input field.
        overwrite_x = self.cutout is not None
        axes = tuple(range(-self.input_grid.ndim, 0))

        f = _fft_module.fftn(f, axes=axes, overwrite_x=overwrite_x)

        if (self._transfer_function.ndim - self.internal_grid.ndim) == 2:
            # The transfer function is a matrix field.
            s1 = f.shape[:-self.internal_grid.ndim] + (self.internal_grid.size,)
            f = Field(f.reshape(s1), self.internal_grid)

            s2 = self._transfer_function.shape[:-self.internal_grid.ndim] + (self.internal_grid.size,)
            tf = Field(self._transfer_function.reshape(s2), self.internal_grid)

            if adjoint:
                tf = field_conjugate_transpose(tf)

            f = field_dot(tf, f).shaped
        else:
            # The transfer function is a scalar field.
            if adjoint:
                tf = self._transfer_function.conj()
            else:
                tf = self._transfer_function

            # This is faster than f *= tf for Numpy due to it going back to C ordering.
            f = f * tf

        # Since f is now guaranteed to not share memory, always allow overwriting.
        overwrite_x = True

        f = _fft_module.ifftn(f, axes=axes, overwrite_x=overwrite_x)

        s = f.shape[:-self.internal_grid.ndim] + (-1,)
        if self.cutout is None:
            res = f.reshape(s)
        else:
            res = f[c].reshape(s)

        return Field(res, self.input_grid)
