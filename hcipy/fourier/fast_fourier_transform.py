from __future__ import division

import numpy as np
from .fourier_transform import FourierTransform, multiplex_for_tensor_fields
from ..field import Field, CartesianGrid, RegularCoords

def make_fft_grid(input_grid, q=1, fov=1):
	q = np.ones(input_grid.ndim, dtype='float') * q
	fov = np.ones(input_grid.ndim, dtype='float') * fov

	# Check assumptions
	if not input_grid.is_regular:
		raise ValueError('The input_grid must be regular.')
	if not input_grid.is_('cartesian'):
		raise ValueError('The input_grid must be cartesian.')

	delta = (2*np.pi / (input_grid.delta * input_grid.dims)) / q
	dims = (input_grid.dims * fov * q).astype('int')
	zero = delta * (-dims/2 + np.mod(dims, 2) * 0.5)

	return CartesianGrid(RegularCoords(delta, dims, zero))

class FastFourierTransform(FourierTransform):
	def __init__(self, input_grid, q=1, fov=1, shift=0):
		# Check assumptions
		if not input_grid.is_regular:
			raise ValueError('The input_grid must be regular.')
		if not input_grid.is_('cartesian'):
			raise ValueError('The input_grid must be cartesian.')

		self.input_grid = input_grid

		self.shape_in = input_grid.shape
		self.weights = input_grid.weights[0]
		self.size = input_grid.size
		self.ndim = input_grid.ndim

		self.output_grid = make_fft_grid(input_grid, q, fov).shifted(shift)

		self.shape_out = self.output_grid.shape
		self.internal_shape = (self.shape_in * q).astype('int')
		self.internal_array = np.zeros(self.internal_shape, 'complex')

		if np.allclose(self.internal_shape, self.shape_in):
			self.cutout_input = None
		else:
			cutout_start = (self.internal_shape / 2.).astype('int') - (self.shape_in / 2.).astype('int')
			cutout_end = cutout_start + self.shape_in
			self.cutout_input = tuple([slice(start, end) for start, end in zip(cutout_start, cutout_end)])

		if np.allclose(self.internal_shape, self.shape_out):
			self.cutout_output = None
		else:
			cutout_start = (self.internal_shape / 2.).astype('int') - (self.shape_out / 2.).astype('int')
			cutout_end = cutout_start + self.shape_out
			self.cutout_output = tuple([slice(start, end) for start, end in zip(cutout_start, cutout_end)])

		center = input_grid.zero + input_grid.delta * (np.array(input_grid.dims) // 2) - input_grid.delta * (self.internal_shape // 2)

		self.shift_input = np.exp(-1j * np.dot(center, self.output_grid.coords))
		self.shift_input /= np.fft.ifftshift(self.shift_input.reshape(self.shape_out)).ravel()[0] # No piston shift (remove central shift phase)
		if np.allclose(self.shift_input, 1):
			self.shift_input = None

		shift = np.ones(self.input_grid.ndim) * (shift - self.output_grid.delta * (np.array(self.output_grid.dims) // 2))
		self.shift_output = np.exp(-1j * np.dot(shift, self.input_grid.coords)).reshape(self.shape_in)
		if np.allclose(self.shift_output, 1):
			self.shift_output = None

	@multiplex_for_tensor_fields
	def forward(self, field):
		if self.cutout_input is None:
			if self.shift_output is None:
				f = field.reshape(self.shape_in)
			else:
				self.internal_array[:] = field.reshape(self.shape_in)
				self.internal_array *= self.shift_output

				f = self.internal_array
		else:
			self.internal_array[self.cutout_input] = field.reshape(self.shape_in)
			if self.shift_output is not None:
				self.internal_array[self.cutout_input] *= self.shift_output.reshape(self.shape_in)

			f = self.internal_array

		fft_array = np.fft.fftn(f)

		if self.cutout_output is None:
			res = fft_array.ravel()
		else:
			res = fft_array[self.cutout_output].ravel()

		if self.shift_input is not None:
			res *= self.shift_input

		res *= self.weights

		return Field(res, self.output_grid).astype(field.dtype, copy=False)

	@multiplex_for_tensor_fields
	def backward(self, field):
		f = np.zeros(self.internal_shape, dtype='complex')
		f[self.cutout_output] = (field.ravel() / self.shift_input).reshape(self.shape_out)
		res = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(f)))
		res = res[self.cutout_input].ravel() / self.weights / self.shift_output

		return Field(res, self.input_grid).astype(field.dtype)
