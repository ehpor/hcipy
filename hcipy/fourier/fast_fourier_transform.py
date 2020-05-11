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
	def __init__(self, input_grid, q=1, fov=1, shift=0, high_accuracy=False):
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
		self.high_accuracy = high_accuracy

		self.output_grid = make_fft_grid(input_grid, q, fov).shifted(shift)
		self.internal_grid = make_fft_grid(input_grid, q, 1)

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

		center = input_grid.zero + input_grid.delta * (np.array(input_grid.dims) // 2)
		self.shift_input = np.exp(-1j * np.dot(center, self.output_grid.coords))
		self.shift_input /= np.fft.ifftshift(self.shift_input.reshape(self.shape_out)).ravel()[0] # No piston shift (remove central shift phase)

		if not high_accuracy:
			fftshift = input_grid.delta * (np.array(self.internal_shape[::-1] // 2))
			fftshift = np.exp(1j * (np.dot(fftshift, self.internal_grid.coords))).reshape(self.internal_shape)

			if self.cutout_output:
				self.shift_input *= fftshift[self.cutout_output].ravel()
			else:
				self.shift_input *= fftshift.ravel()

		self.shift_input *= self.weights

		shift = np.ones(self.input_grid.ndim) * shift
		self.shift_output = np.exp(-1j * np.dot(shift, self.input_grid.coords))

		if not high_accuracy:
			fftshift = self.input_grid.delta * (np.array(self.internal_shape[::-1] // 2))
			fftshift = np.exp(1j * (np.dot(fftshift, self.internal_grid.coords) - np.dot(fftshift, self.internal_grid.zero))).reshape(self.internal_shape)

			if self.cutout_input:
				self.shift_output *= fftshift[self.cutout_input].ravel()
			else:
				self.shift_output *= fftshift.ravel()

	@multiplex_for_tensor_fields
	def forward(self, field):
		if self.cutout_input is None:
			self.internal_array[:] = field.reshape(self.shape_in)
			self.internal_array *= self.shift_output.reshape(self.shape_in)
		else:
			self.internal_array[:] = 0
			self.internal_array[self.cutout_input] = field.reshape(self.shape_in)
			self.internal_array[self.cutout_input] *= self.shift_output.reshape(self.shape_in)

		if self.high_accuracy:
			self.internal_array = np.fft.ifftshift(self.internal_array)

		fft_array = np.fft.fftn(self.internal_array)

		if self.high_accuracy:
			fft_array = np.fft.fftshift(fft_array)

		if self.cutout_output is None:
			res = fft_array.ravel()
		else:
			res = fft_array[self.cutout_output].ravel()

		res *= self.shift_input

		return Field(res, self.output_grid).astype(field.dtype, copy=False)

	@multiplex_for_tensor_fields
	def backward(self, field):
		if self.cutout_output is None:
			self.internal_array[:] = field.reshape(self.shape_out)
			self.internal_array /= self.shift_input.reshape(self.shape_out)
		else:
			self.internal_array[:] = 0
			self.internal_array[self.cutout_output] = field.reshape(self.shape_out)
			self.internal_array[self.cutout_output] /= self.shift_input.reshape(self.shape_out)

		if self.high_accuracy:
			self.internal_array = np.fft.ifftshift(self.internal_array)

		fft_array = np.fft.ifftn(self.internal_array)

		if self.high_accuracy:
			fft_array = np.fft.fftshift(fft_array)

		if self.cutout_input is None:
			res = fft_array.ravel()
		else:
			res = fft_array[self.cutout_input].ravel()

		res /= self.shift_output

		return Field(res, self.input_grid).astype(field.dtype, copy=False)
