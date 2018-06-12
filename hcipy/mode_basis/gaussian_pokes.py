
from ..field import Field
from .mode_basis import ModeBasis
import numpy as np

def make_gaussian_pokes(grid, mu, sigma):
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
	
	Returns
	-------
	ModeBasis
		The calculated mode basis.
	'''
	sigma = np.ones(mu.size) * sigma
	return ModeBasis([Field(np.exp(-0.5*grid.shifted(p).as_('polar').r**2/s**2), grid) for p, s in zip(mu.points, sigma)])
