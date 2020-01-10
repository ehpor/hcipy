import numpy as np
from ..field import Field

def large_poisson(lam, thresh=1e6):
	"""
	Draw samples from a Poisson distribution, taking care of large values of `lam`.

	At large values of `lam` the distribution automatically switches to the corresponding normal distribution.
	This switch is independently decided for each expectation value in the `lam` array.

	Parameters
	----------
	lam : array_like
		Expectation value for the Poisson distribution. Must be >= 0.
	thresh : float
		The threshold at which the distribution switched from a Poisson to a normal distribution.
	
	Returns
	-------
	array_like
		The drawn samples from the Poisson or normal distribution, depending on the expectation value.
	"""
	large = lam > thresh
	small = ~large
	
	# Use normal approximation if the number of photons is large
	n = np.zeros(lam.shape)
	n[large] = np.round(lam[large] + np.random.normal(size=np.sum(large)) * np.sqrt(lam[large]))
	n[small] = np.random.poisson(lam[small], size=np.sum(small))

	if hasattr(lam, 'grid'):
		n = Field(n, lam.grid)
	
	return n