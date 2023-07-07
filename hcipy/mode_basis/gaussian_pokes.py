from ..field import evaluate_supersampled
import numpy as np
from scipy.sparse import csr_matrix

def make_gaussian_pokes(grid, mu, sigma, cutoff=5):
    '''Make a basis of Gaussians.

    Parameters
    ----------
    grid : Grid or None
        The grid on which to calculate the mode basis. If this is None, a list of Field
        generators will be returned instead of a mode basis.
    mu : Grid
        The centers for each of the Gaussians.
    sigma : ndarray or scalar
        The standard deviation of each of the Gaussians. If this is a scalar,
        this value will be used for all Gaussians.
    cutoff : scalar or None
        The factor of sigma beyond which the Gaussian will be set to zero. The ModeBasis will
        be sparse to reduce memory usage. If the cutoff is None, there will be no cutoff, and
        the returned ModeBasis will be dense.

    Returns
    -------
    ModeBasis or list of Field generators.
        The sparse mode basis. If cutoff is None, a dense mode basis will be returned.
    '''
    sigma = np.ones(mu.size) * sigma

    def poke(m, s):
        def eval_func(func_grid):
            if func_grid.is_('cartesian'):
                r2 = (func_grid.x - m[0])**2 + (func_grid.y - m[1])**2
            else:
                r2 = func_grid.shifted(-m).as_('polar').r**2

            res = np.exp(-0.5 * r2 / s**2)

            if cutoff is not None:
                res -= np.exp(-0.5 * cutoff**2)
                res[r2 > (cutoff * s)**2] = 0

                res = csr_matrix(res)
                res.eliminate_zeros()

            return res
        return eval_func

    pokes = [poke(m, s) for m, s in zip(mu.points, sigma)]

    if grid is None:
        return pokes
    else:
        return evaluate_supersampled(pokes, grid, 1, make_sparse=True)
