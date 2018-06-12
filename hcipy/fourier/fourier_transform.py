import numpy as np
from ..field import Field

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
		--------
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
		--------
		Field
			The inverse Fourier transform of the field.
		'''
		raise NotImplementedError()
	
	def get_transformation_matrix_forward(self):
		'''Returns the transformation matrix corresonding to the
		Fourier transform.

		Returns
		--------
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
		--------
		ndarray
			A matrix representing the Fourier transform.
		'''
		coords_in = self.input_grid.as_('cartesian').coords
		coords_out = self.output_grid.as_('cartesian').coords

		A = np.exp(1j * np.dot(np.array(coords_in).T, coords_out))
		A *= self.output_grid.weights
		A /= (2*np.pi)**self.input_grid.ndim

		return A

def time_it(function, t_max=5, n_max=100):
	import time
	
	start = time.time()
	times = []
	
	while (time.time() < start + t_max) and (len(times) < n_max):
		t1 = time.time()
		function()
		t2 = time.time()
		times.append(t2 - t1)
	
	return np.median(times)

def make_fourier_transform(input_grid, output_grid=None, q=1, fov=1, planner='estimate'):
	'''Construct a FourierTransform object.

	The most time-efficient Fourier transform method will be chosen according to actual or estimated performance.

	Parameters
	----------
	input_grid : Grid
		The grid that will be used for the Field passed to the Fourier transform.
	output_grid : None or Grid
		The grid of the resulting field. If it is None, a optimal grid will be chosen, according to `q` and `fov`.
	q : scalar
		Describes how many samples to take in the Fourier domain. A value of 1 means critcally sampled in the Fourier domain.
	fov : scalar
		Describes how far out the Fourier domain extends. A value of 1 means the same amount of samples as the spatial domain.
	planner : string
		If it is 'estimate', performance of the different methods will be estimated from theoretical complexity estimates. 
		If it is 'measure', actual Fourier transforms will be performed to get the actual performance. The latter takes longer,
		but is more accurate.
	
	Returns
	-------
	FourierTransform
		The Fourier transform that was requested.
	'''
	from .fast_fourier_transform import FastFourierTransform, make_fft_grid
	from .matrix_fourier_transform import MatrixFourierTransform
	from .naive_fourier_transform import NaiveFourierTransform

	if output_grid is None:
		# Choose between FFT and MFT
		if not (input_grid.is_regular and input_grid.is_('cartesian')):
			raise ValueError('For non-regular non-cartesian Grids, a Fourier transform is required to have an output_grid.')

		if input_grid.ndim not in [1,2]:
			method = 'fft'
		else:
			output_grid = make_fft_grid(input_grid, q, fov)

			if planner == 'estimate':
				# Estimate analytically from complexities
				N_in = input_grid.shape * q
				N_out = output_grid.shape

				if input_grid.ndim == 1:
					fft = 4 * N_in[0] * np.log2(N_in)
					mft = 4 * input_grid.size * N_out[0]
				else:
					fft = 4 * np.prod(N_in) * np.log2(np.prod(N_in))
					mft = 4 * (np.prod(input_grid.shape) * N_out[1] + np.prod(N_out) * input_grid.shape[0])
				if fft > mft:
					method = 'mft'
				else:
					method = 'fft'
			elif planner == 'measure':
				# Measure directly
				fft = FastFourierTransform(input_grid, q, fov)
				mft = MatrixFourierTransform(input_grid, output_grid)

				a = np.zeros(input_grid.size, dtype='complex')
				fft_time = time_it(lambda: fft.forward(a))
				mft_time = time_it(lambda: mft.forward(a))

				if fft_time > mft_time:
					method = 'mft'
				else:
					method = 'fft'
	else:
		# Choose between MFT and Naive
		if input_grid.is_separated and input_grid.is_('cartesian') and output_grid.is_separated and output_grid.is_('cartesian') and input_grid.ndim in [1,2]:
			method = 'mft'
		else:
			method = 'naive'
	
	# Make the Fourier transform
	if method == 'fft':
		return FastFourierTransform(input_grid, q, fov)
	elif method == 'mft':
		return MatrixFourierTransform(input_grid, output_grid)
	elif method == 'naive':
		return NaiveFourierTransform(input_grid, output_grid)

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
			f = field.reshape((-1,field.grid.size))
			res = [func(self, ff) for ff in f]
			new_shape = np.concatenate((field.tensor_shape, [-1]))
			return Field(np.array(res).reshape(new_shape), res[0].grid)

	return inner