from .mode_basis import ModeBasis
from ..field import evaluate_supersampled
import numpy as np
from scipy.sparse import csr_matrix

def make_gaussian_pokes(grid, mu, sigma, cutoff=5, oversampling=None):
	'''Make a basis of Gaussians.

	Parameters
	----------
	grid : Grid
		The grid on which to calculate the mode basis.
	mu : Grid
		The centers for each of the Gaussians.
	sigma : ndarray or scalar
		The standard deviation of each of the Gaussians. If this is a scalar,
		this value will be used for all Gaussians.
	cutoff : scalar or None
		The factor of sigma beyond which the Gaussian will be set to zero. The ModeBasis will
		be sparse to reduce memory usage. If the cutoff is None, there will be no cutoff, and
		the returned ModeBasis will be dense.
	oversampling : integer or None
		The oversamping factor when creating the Gaussian. If None the Gaussian will be evaluated at grid resolution.

	Returns
	-------
	ModeBasis
		The sparse mode basis. If cutoff is None, a dense mode basis will be returned.
	'''
	sigma = np.ones(mu.size) * sigma

	def poke(m, s):
		if grid.is_('cartesian'):
			r2 = (grid.x - m[0])**2 + (grid.y - m[1])**2
		else:
			r2 = grid.shifted(-m).as_('polar').r**2

		res = np.exp(-0.5 * r2 / s**2)

		if cutoff is not None:
			res -= np.exp(-0.5 * cutoff**2)
			res[r2 > (cutoff * s)**2] = 0

			res = csr_matrix(res)
			res.eliminate_zeros()

		return res

	if oversampling is not None:
		pokes = [evaluate_supersampled(poke(m, s), grid, oversampling) for m, s in zip(mu.points, sigma)]
	else:
		pokes = [poke(m, s)(grid) for m, s in zip(mu.points, sigma)]

	return ModeBasis(pokes, grid)
