
from ..field import Field
from .mode_basis import ModeBasis
import numpy as np

def make_gaussian_pokes(grid, mu, sigma):
	sigma = np.ones(mu.size) * sigma
	return ModeBasis([Field(np.exp(-0.5*grid.shifted(p).as_('polar').r**2/s**2), grid) for p, s in zip(mu.points, sigma)])
