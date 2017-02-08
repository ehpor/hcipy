import numpy as np

def make_pupil_grid(N, D):
	from .cartesian_grid import CartesianGrid
	from .coordinates import RegularCoords

	D = (np.ones(2) * D).astype('float')
	N = (np.ones(2) * N).astype('int')

	delta = D / (N-1)
	zero = -D/2

	return CartesianGrid(RegularCoords(delta, N, zero))

def make_focal_grid(pupil_grid, q=1, fov=1, focal_length=1, wavelength=1):
	from ..fourier import make_fft_grid

	f_lambda = focal_length * wavelength
	uv = make_fft_grid(input_grid, q, fov)

	focal_grid = uv * (f_lambda_ref / (2*np.pi))
	
	return focal_grid