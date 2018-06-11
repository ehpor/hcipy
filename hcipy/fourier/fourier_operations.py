import numpy as np

from .fast_fourier_transform import FastFourierTransform
from .matrix_fourier_transform import MatrixFourierTransform

class ConvolveFFT(object):
	def __init__(self, input_grid, kernel):
		self.input_grid = input_grid
		self.kernel = kernel

		fft = FastFourierTransform(input_grid, 2)
		self.internal_shape = fft.internal_shape
		self.cutout = fft.cutout_input

		mft = MatrixFourierTransform(kernel.grid, fft.output_grid)
		

	def forward(self, field):
		f = np.zeros(self.internal_shape, dtype='complex')

class ShearFFT(object):
	pass

class RotateFFT(object):
	pass

class FilterFFT(object):
	pass