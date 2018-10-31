from __future__ import division

import numpy as np
from .fourier_transform import FourierTransform, multiplex_for_tensor_fields

def make_fft_grid(input_grid, q=1, fov=1):
	from ..field import CartesianGrid, RegularCoords

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
		self.weights = input_grid.weights
		self.size = input_grid.size
		self.ndim = input_grid.ndim
		
		self.output_grid = make_fft_grid(input_grid, q, fov).shifted(shift)
		
		self.shape_out = self.output_grid.shape
		self.internal_shape = (self.shape_in * q).astype('int')
		
		cutout_start = (self.internal_shape / 2.).astype('int') - (self.shape_in / 2.).astype('int')
		cutout_end = cutout_start + self.shape_in
		self.cutout_input = [slice(start, end) for start, end in zip(cutout_start, cutout_end)]
		
		cutout_start = (self.internal_shape / 2.).astype('int') - (self.shape_out / 2.).astype('int')
		cutout_end = cutout_start + self.shape_out
		self.cutout_output = [slice(start, end) for start, end in zip(cutout_start, cutout_end)]
		
		center = input_grid.zero + input_grid.delta * (np.array(input_grid.dims) // 2)
		if np.allclose(center, 0):
			self.shift_input = 1
		else:
			self.shift_input = np.exp(-1j * np.dot(center, self.output_grid.coords))
			self.shift_input /= np.fft.ifftshift(self.shift_input.reshape(self.shape_out)).ravel()[0] # No piston shift (remove central shift phase)

		shift = np.ones(self.input_grid.ndim) * shift
		if np.allclose(shift, 0):
			self.shift_output = 1
		else:
			self.shift_output = np.exp(-1j * np.dot(shift, self.input_grid.coords))
	
	@multiplex_for_tensor_fields
	def forward(self, field):
		from ..field import Field
		
		f = np.zeros(self.internal_shape, dtype='complex')
		f[tuple(self.cutout_input)] = (field.ravel() * self.weights * self.shift_output).reshape(self.shape_in)
		res = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(f)))
		res = res[tuple(self.cutout_output)].ravel() * self.shift_input
		
		return Field(res, self.output_grid)
	
	@multiplex_for_tensor_fields
	def backward(self, field):
		from ..field import Field

		f = np.zeros(self.internal_shape, dtype='complex')
		f[tuple(self.cutout_output)] = (field.ravel() / self.shift_input).reshape(self.shape_out)
		res = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(f)))
		res = res[tuple(self.cutout_input)].ravel() / self.weights / self.shift_output
		
		return Field(res, self.input_grid)
