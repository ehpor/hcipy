import numpy as np

from .coordinates import UnstructuredCoords
from .field import Field
from .grid import Grid
from .._math.backends import is_scalar, infer_xp

from functools import reduce
import operator

def _prod(iterable):
    return reduce(operator.mul, iterable, 1)

def _get_rotation_matrix(ndim, angle, axis=None, xp=None):
    if xp is None:
        xp = infer_xp(angle, axis)

    if ndim == 1:
        raise ValueError('Rotation of a one-dimensional grid is not possible.')
    elif ndim > 3:
        raise NotImplementedError()

    # Convert to array if scalar.
    angle = xp.asarray(angle)

    if ndim == 2:
        cos_a = xp.cos(angle)
        sin_a = xp.sin(angle)

        return xp.stack([
            xp.stack([cos_a, -sin_a]),
            xp.stack([sin_a, cos_a]),
        ])

    elif ndim == 3:
        if axis is None:
            raise ValueError('An axis must be supplied when rotating a three-dimensional grid.')

        axis = xp.asarray(axis)

        if axis.shape != (3,):
            raise ValueError('The axis must be a 3-vector.')

        axis = axis / xp.sqrt(xp.vecdot(axis, axis))

        K = xp.stack([
            xp.stack([ 0, -axis[2], axis[1]]),
            xp.stack([ axis[2], 0, -axis[0]]),
            xp.stack([-axis[1], axis[0], 0]),
        ])

        return xp.eye(3, dtype=K.dtype) + xp.sin(angle) * K + (1 - xp.cos(angle)) * (K @ K)

class CartesianGrid(Grid):
    '''A grid representing a N-dimensional Cartesian coordinate system.
    '''
    _coordinate_system = 'cartesian'

    @property
    def x(self):
        '''The x-coordinate (dimension 0).
        '''
        return Field(self.coords[0], self)

    @property
    def y(self):
        '''The y-coordinate (dimension 1).
        '''
        return Field(self.coords[1], self)

    @property
    def z(self):
        '''The z-coordinate (dimension 2).
        '''
        return Field(self.coords[2], self)

    @property
    def w(self):
        '''The w-coordinate (dimension 3).
        '''
        return Field(self.coords[3], self)

    def scale(self, scale):
        '''Scale the grid in-place.

        Parameters
        ----------
        scale : array_like or scalar
            The factor with which to scale the grid.

        Returns
        -------
        Grid
            Itself to allow for chaining these transformations.
        '''
        if is_scalar(scale):
            self.weights *= self.xp.abs(self.xp.asarray(scale))**self.ndim
        else:
            self.weights *= self.xp.prod(self.xp.abs(scale))

        self.coords *= scale

        return self

    def shift(self, shift):
        '''Shift the grid in-place.

        Parameters
        ----------
        shift : array_like
            The amount with which to shift the grid.

        Returns
        -------
        Grid
            Itself to allow for chaining these transformations.
        '''
        self.coords += shift
        return self

    def rotate(self, angle, axis=None):
        '''Rotate the grid in-place.

        .. caution::
            All structure in the coordinates will be destroyed.

        Parameters
        ----------
        angle : scalar
            The angle in radians.
        axis : ndarray or None
            The axis of rotation. For two-dimensional grids, it is ignored. For
            three-dimensional grids it is required.

        Returns
        -------
        Grid
            Itself to allow for chaining these transformations.
        '''
        xp = self.coords.xp
        R = _get_rotation_matrix(self.ndim, angle, axis, xp=xp)

        coords = xp.einsum('ik,kn->in', R, xp.asarray(self.coords))
        self.coords = UnstructuredCoords(list(coords))
        return self

    def rotated(self, angle, axis=None):
        '''A rotated copy of this grid.

        Parameters
        ----------
        angle : scalar
            The angle in radians.
        axis : ndarray or None
            The axis of rotation. For two-dimensional grids, it is ignored. For
            three-dimensional grids it is required.

        Returns
        -------
        Grid
            The rotated grid.
        '''
        xp = self.coords.xp
        R = _get_rotation_matrix(self.ndim, angle, axis, xp=xp)

        coords = xp.einsum('ik,kn->in', R, xp.stack(list(self.coords)))
        return CartesianGrid(UnstructuredCoords(list(coords)))

    @staticmethod
    def _get_automatic_weights(coords):
        xp = coords.xp

        if coords.is_regular:
            return xp.prod(coords.delta)
        elif coords.is_separated:
            weights = None
            n = coords.ndim
            for i, x in enumerate(coords.separated_coords):
                w = (x[2:] - x[:-2]) / 2
                w = xp.concat([
                    xp.reshape(x[1] - x[0], (1,)),
                    w,
                    xp.reshape(x[-1] - x[-2], (1,)),
                ])

                shape = (1,) * (n - i - 1) + (-1,) + (1,) * i
                w = xp.reshape(w, shape)

                weights = w if weights is None else weights * w

            return xp.reshape(weights, (-1,))

Grid._add_coordinate_system('cartesian', CartesianGrid)
