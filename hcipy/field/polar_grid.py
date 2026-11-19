import numpy as np

from .field import Field
from .grid import Grid
from .coordinates import UnstructuredCoords
from .._math.backends import array_namespace

def _trapezoid_weights(x, xp):
    w = (x[2:] - x[:-2]) / 2

    return xp.concat([
        xp.reshape(x[1] - x[0], (1,)),
        w,
        xp.reshape(x[-1] - x[-2], (1,)),
    ])

class PolarGrid(Grid):
    '''A grid representing a two-dimensional Polar coordinate system.
    '''
    _coordinate_system = 'polar'

    @property
    def r(self):
        '''The radial coordinate (dimension 0).
        '''
        return Field(self.coords[0], self)

    @property
    def theta(self):
        ''' The angular coordinate (dimension 1).
        '''
        return Field(self.coords[1], self)

    def scale(self, scale):
        '''Scale the grid in-place.

        Parameters
        ----------
        scale : array_like
            The factor with which to scale the grid.

        Returns
        -------
        Grid
            Itself to allow for chaining these transformations.
        '''
        xp = self.coords.xp
        self.coords *= xp.asarray([scale, 1.0])
        self.weights *= np.abs(scale)**(self.ndim)
        return self

    def shift(self, shift):
        '''Shift the grid in-place.

        .. caution::
            All structure in the coordinates will be destroyed.

        Parameters
        ----------
        shift : array_like
            The amount with which to shift the grid.

        Returns
        -------
        Grid
            Itself to allow for chaining these transformations.
        '''
        new_grid = PolarGrid(UnstructuredCoords(self.coords), self.weights).as_('cartesian')
        self.coords = new_grid.shifted(shift).as_('polar').coords

        return self

    def shifted(self, shift):
        '''A shifted copy of this grid.

        .. caution::
            The returned grid is a Cartesian grid.

        Parameters
        ----------
        shift : array_like
            The amount with which to shift the grid.

        Returns
        -------
        Grid
            The scaled grid.
        '''
        grid = self.as_('cartesian')
        grid.shift(shift)
        return grid

    def rotate(self, angle, axis=None):
        '''Rotate the grid in-place.

        Parameters
        ----------
        angle : scalar
            The angle in radians.
        axis : ndarray or None
            The axis of rotation. For this (polar) grid, it is ignored.

        Returns
        -------
        Grid
            Itself to allow for chaining these transformations.
        '''
        self.coords += (0, angle)
        return self

    @staticmethod
    def _get_automatic_weights(coords):
        '''Calculate automatic area weights for a polar grid.

        Each point is weighted by the polar area element :math:`r\\,dr\\,d\\theta`.
        The cell width along each coordinate is half the distance to the previous
        sample plus half the distance to the next sample. The first and last samples
        only have one neighbour, so they use the full distance to that neighbour.
        '''
        # Unstructured or non-2D polar grids have no unambiguous area element.
        if not coords.is_separated or coords.ndim != 2:
            return None

        xp = coords.xp
        r, theta = coords.separated_coords

        if r.shape[0] < 2 or theta.shape[0] < 2:
            return None

        radial_widths = _trapezoid_weights(r, xp)
        angular_widths = _trapezoid_weights(theta, xp)

        weights = xp.reshape(r * xp.abs(radial_widths), (1, -1)) * xp.reshape(xp.abs(angular_widths), (-1, 1))

        return xp.reshape(weights, (-1,))

def _cartesian_to_polar(self):
    from .coordinates import UnstructuredCoords

    x, y = self.coords

    xp = array_namespace(x, y)
    r = xp.hypot(x, y)
    theta = xp.atan2(y, x)

    return PolarGrid(UnstructuredCoords([r, theta], xp=xp))

def _polar_to_cartesian(self):
    from .coordinates import UnstructuredCoords
    from .cartesian_grid import CartesianGrid

    r, theta = self.coords

    xp = array_namespace(r, theta)
    x = r * xp.cos(theta)
    y = r * xp.sin(theta)

    return CartesianGrid(UnstructuredCoords([x, y], xp=xp))

Grid._add_coordinate_system('polar', PolarGrid)

Grid._add_coordinate_system_transformation('cartesian', 'polar', _cartesian_to_polar)
Grid._add_coordinate_system_transformation('polar', 'cartesian', _polar_to_cartesian)
