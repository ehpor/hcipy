import numpy as np

def make_fft_grid(input_grid, q=1, fov=1):
	from ..field import CartesianGrid, RegularCoords
	
	# Check assumptions
	if not input_grid.is_regular:
		raise ValueError('The input_grid must be regular.')
	if not input_grid.is_('cartesian'):
		raise ValueError('The input_grid must be cartesian.')
	
	delta = (2*np.pi / (input_grid.delta * input_grid.shape)) / q
	shape = (input_grd.shape * fov * q).astype('int')
	zero = delta * (-shape/2).astype('int')
	
	return CartesianGrid(RegularCoords(delta, shape, zero))

class FastFourierTransform(object):
	def __init__(self, input_grid, q=1, fov=1):
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
		
		self.output_grid = make_fft_grid(input_grid, q, fov)
		
		self.shape_out = output_grid.shape
		self.internal_shape = (self.shape_in * q).astype('int')
		
		cutout_start = (self.internal_shape / 2.).astype('int') - (self.shape_in / 2.).astype('int')
		cutout_end = cutout_start + self.shape_in
		self.cutout_input = [slice(start, end) for start, end in zip(cutout_start, cutout_end)]
		
		cutout_start = (self.internal_shape / 2.).astype('int') - (self.shape_out / 2.).astype('int')
		cutout_end = cutout_start + self.shape_out
		self.cutout_output = [slice(start, end) for start, end in zip(cutout_start, cutout_end)]
		
		center = input_grid.zero + input_grid.delta * (np.array(input_grid.shape) // 2)
		
		center = input_grid.zero + input_grid.delta * (np.array(input_grid.shape) // 2)
		if np.allclose(center, 0):
			self.shift = 1
		else:
			self.shift = np.exp(-1j * np.dot(center, self.output_grid.coords))
			self.shift /= np.fft.fftshift(self.shift.reshape(self.shape_out)).ravel()[0] # No piston shift (remove central shift phase)
	
	def __call__(self, field):
		return self.forward(field)
	
	def forward(self, field):
		f = np.zeros(self.internal_shape, dtype='complex')
		f[self.cutout_input] = (field.ravel() * self.weights).reshape(self.shape_in)
		res = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(f)))
		res = res[self.cutout_output].ravel() * self.shift
		
		if hasattr(field, 'grid'):
			from ..field import Field
			return Field(res, self.output_grid)
		else:
			return res
	
	def inverse(self, field):
		f = np.zeros(self.internal_shape, dtype='complex')
		f[self.cutout_output] = (field.ravel() / self.shift).reshape(self.shape_out)
		res = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(f)))
		res = res[self.cutout_input].ravel() / self.weights
		
		if hasattr(field, 'grid'):
			from .sampled_function import SampledFunction
			return SampledFunction(res, self.input_grid)
		else:
			return res
