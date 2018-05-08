import numpy as np

def make_uniform_grid(dims, extent, center=0, has_center=False):
	'''Create a uniformly-spaced :class:`Grid` of a certain shape and size.

	Parameters
	----------
	dims : scalar or ndarray
		The number of points in each dimension. If this is a scalar, it will
		be multiplexed over all dimensions.
	extent : scalar or ndarray
		The total extent of the grid in each dimension.
	center : scalar or ndarray
		The center point. The grid will by symmetric around this point.
	has_center : boolean
		Does the grid has to have the center as one of its points. If this is
		False, this does not mean that the grid will not have the center.

	Returns
	-------
	Grid
		A :class:`Grid` with :class:`RegularCoords`.
	'''
	from .cartesian_grid import CartesianGrid
	from .coordinates import RegularCoords

	num_dims = max(np.array([dims]).shape[-1], np.array([extent]).shape[-1], np.array([center]).shape[-1])

	dims = (np.ones(num_dims) * dims).astype('float')
	extent = (np.ones(num_dims) * extent).astype('int')
	center = (np.ones(num_dims) * center).astype('float')

	delta = extent / dims
	zero = -extent / 2 + center + delta / 2

	if has_center:
		zero -= delta/2 * (1 - np.mod(dims, 2))

	return CartesianGrid(RegularCoords(delta, dims, zero))

def make_pupil_grid(N, D=1):
	from .cartesian_grid import CartesianGrid
	from .coordinates import RegularCoords

	D = (np.ones(2) * D).astype('float')
	N = (np.ones(2) * N).astype('int')

	delta = D / (N-1)
	zero = -D/2

	return CartesianGrid(RegularCoords(delta, N, zero))

def make_focal_grid(pupil_grid, q=1, num_airy=None, focal_length=1, wavelength=1):
	from ..fourier import make_fft_grid

	f_lambda = focal_length * wavelength
	if num_airy is None:
		fov = 1
	else:
		fov = (num_airy * np.ones(pupil_grid.ndim, dtype='float')) / (pupil_grid.shape / 2)
	
	if np.max(fov) > 1:
		import warnings
		warnings.warn('Focal grid is larger than the maximum allowed angle (fov=%.03f). You may see wrapping when doing propagations.' % np.max(fov), stacklevel=2)
		
	uv = make_fft_grid(pupil_grid, q, fov)
	focal_grid = uv.scaled(f_lambda / (2*np.pi))
	
	return focal_grid

def make_hexagonal_grid(circum_diameter, n_rings):
	from .cartesian_grid import CartesianGrid
	from .coordinates import UnstructuredCoords

	apothem = circum_diameter * np.sqrt(3) / 4

	q = [0]
	r = [0]
	
	for n in range(1,n_rings+1):
		#top
		q += list(range(n,0,-1))
		r += list(range(0,n))
		# right top
		q += list(range(0,-n,-1))
		r += [n] * n
		# right bottom
		q += [-n] * n
		r += list(range(n,0,-1))
		# bottom
		q += list(range(-n,0))
		r += list(range(0,-n,-1))
		# left bottom
		q += list(range(0,n))
		r += [-n] * n
		# left top
		q += [n] * n
		r += list(range(-n,0))
	
	x = (-np.array(q) + np.array(r)) * circum_diameter / 2
	y = (np.array(q) + np.array(r)) * apothem * 2

	weight = 2 * apothem**2 * np.sqrt(3)
	
	return CartesianGrid(UnstructuredCoords((x,y)), weight)

def make_chebyshev_grid(dims, minimum=None, maximum=None):
	from .cartesian_grid import CartesianGrid
	from .coordinates import SeparatedCoords

	if minimum is None:
		minimum = -1
	
	if maximum is None:
		maximum = 1
	
	dims = np.array(dims)
	minimum = np.ones(len(dims)) * minimum
	maximum = np.ones(len(dims)) * maximum

	middles = (minimum + maximum) / 2
	intervals = (maximum - minimum) / 2

	sep_coords = []
	for dim, middle, interval in zip(dims, middles, intervals):
		c = np.cos(np.pi * (2 * np.arange(dim) + 1) / (2.0 * dim))
		c = middle + interval * c
		sep_coords.append(c)
	
	return CartesianGrid(SeparatedCoords(sep_coords))

def subsample_field(field, subsampling):
	pass

def evaluate_supersampled(field_generator, grid, oversampling):
	pass