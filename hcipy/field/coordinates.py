import numpy as np
import copy
import math
from collections.abc import Iterable
import array_api_compat
from .backends import _infer_xp


class Coords(object):
    '''Base class for coordinates.
    '''
    _coordinate_types = {}

    @property
    def xp(self):
        '''The array namespace (e.g., numpy, cupy, jax.numpy) used by these coordinates.
        '''
        return self._xp

    def __copy__(self):
        '''Make a shallow copy of the coordinates.

        This method is required because the Coords stores a reference to a module
        (the xp/backend), and modules cannot be copied. The xp module is shared
        between the original and the copy.

        Returns
        -------
        Coords
            A shallow copy of the coordinates.
        '''
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        '''Make a deep copy of the coordinates.

        This method is required because the Coords stores a reference to a module
        (the xp/backend), and modules cannot be deep copied. The xp module is shared
        between the original and the copy, while all other attributes are deep copied.

        Parameters
        ----------
        memo : dict
            A dictionary to store objects that have already been copied,
            used to handle circular references.

        Returns
        -------
        Coords
            A deep copy of the coordinates.
        '''
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        # Copy all attributes except _xp
        for k, v in self.__dict__.items():
            if k == '_xp':
                # Share the xp module (modules can't be deep copied)
                setattr(result, k, v)
            else:
                setattr(result, k, copy.deepcopy(v, memo))

        return result

    def copy(self):
        '''Make a copy.
        '''
        return copy.deepcopy(self)

    @classmethod
    def from_dict(cls, tree):
        coordinate_class = Coords._coordinate_types[tree['type']]

        return coordinate_class.from_dict(tree)

    def to_dict(self):
        raise NotImplementedError()

    def __add__(self, b):
        '''Add `b` to the coordinates separately and return the result.
        '''
        res = self.copy()
        res += b
        return res

    def __iadd__(self, b):
        '''Add `b` to the coordinates separately in-place.
        '''
        raise NotImplementedError()

    def __radd__(self, b):
        '''Add `b` to the coordinates separately and return the result.
        '''
        return self + b

    def __sub__(self, b):
        '''Subtract `b` from the coordinates separately and return the result.
        '''
        res = self.copy()
        res -= b
        return res

    def __isub__(self, b):
        '''Subtract `b` from the coordinates separately in-place.
        '''
        raise NotImplementedError()

    def __mul__(self, f):
        '''Multiply each coordinate with `f` separately and return the result.
        '''
        res = self.copy()
        res *= f
        return res

    def __imul__(self, f):
        '''Multiply each coordinate with `f` separately in-place.
        '''
        raise NotImplementedError()

    def __rmul__(self, f):
        '''Multiply each coordinate with `f` separately and return the result.
        '''
        return self * f

    def __truediv__(self, f):
        '''Divide each coordinate by `f` separately and return the result.
        '''
        return self * (1 / f)

    def __itruediv__(self, f):
        '''Divide each coordinate by `f` separately in-place.
        '''
        self *= 1 / f
        return self

    def __eq__(self, other):
        '''Check if the coordinates are identical.

        Parameters
        ----------
        other : object
            The object to which to compare.

        Returns
        -------
        boolean
            Whether the two objects are identical.
        '''
        return NotImplemented

    def __getitem__(self, i):
        '''The `i`-th point for these coordinates.
        '''
        raise NotImplementedError()

    @property
    def is_separated(self):
        '''True if the coordinates are separated, False otherwise.
        '''
        return hasattr(self, 'separated_coords')

    @property
    def is_regular(self):
        '''True if the coordinates are regularly-spaced, False otherwise.
        '''
        return hasattr(self, 'regular_coords')

    @property
    def is_unstructured(self):
        '''True if the coordinates are not structured, False otherwise.
        '''
        return not self.is_separated

    def reverse(self):
        '''Reverse the ordering of points in-place.
        '''
        raise NotImplementedError()

    @property
    def size(self):
        '''The number of points.
        '''
        raise NotImplementedError()

    def __len__(self):
        '''The number of dimensions.
        '''
        raise NotImplementedError()

    @property
    def ndim(self):
        '''The number of dimensions.
        '''
        return len(self)

    @classmethod
    def _add_coordinate_type(cls, coordinate_type, coordinate_class):
        cls._coordinate_types[coordinate_type] = coordinate_class


class UnstructuredCoords(Coords):
    '''An unstructured list of points.

    Parameters
    ----------
    coords : list or tuple
        A tuple of a list of positions for each dimension.
    xp : module, optional
        The array namespace (e.g., numpy, cupy, jax.numpy). If not provided,
        it will be inferred from the input arrays.
    '''
    def __init__(self, coords, xp=None):
        if xp is None:
            xp = _infer_xp(*coords)
        self._xp = xp
        self.coords = [self._xp.asarray(c) for c in coords]

    @classmethod
    def from_dict(cls, tree):
        '''Make an UnstructuredCoords from a dictionary, previously created by `to_dict()`.

        Parameters
        ----------
        tree : dictionary
            The dictionary from which to make a new UnstructuredCoords object.

        Returns
        -------
        UnstructuredCoords
            The created object.

        Raises
        ------
        ValueError
            If the dictionary is not formatted correctly.
        '''
        if tree['type'] != 'unstructured':
            raise ValueError('The type of coordinates should be "unstructured".')

        return cls(tree['coords'])

    def to_dict(self):
        '''Convert the object to a dictionary for serialization.

        Returns
        -------
        dictionary
            The created dictionary.
        '''
        tree = {
            'type': 'unstructured',
            'coords': self.coords
        }

        return tree

    def __getstate__(self):
        '''Get state for pickling.
        '''
        return {
            'coords': self.coords,
            'xp_name': self._xp.__name__
        }

    def __setstate__(self, state):
        '''Restore state from pickle.
        '''
        self.coords = state['coords']
        # Restore xp from module name
        if state['xp_name'] == 'numpy':
            self._xp = np
        else:
            # For other backends, import dynamically
            import importlib
            self._xp = importlib.import_module(state['xp_name'])

    @property
    def size(self):
        '''The number of points.
        '''
        return self.coords[0].size

    def __len__(self):
        '''The number of dimensions.
        '''
        return len(self.coords)

    def __getitem__(self, i):
        '''The `i`-th point for these coordinates.
        '''
        return self.coords[i]

    def __iadd__(self, b):
        '''Add `b` to the coordinates separately in-place.
        '''
        if not isinstance(b, Iterable):
            b = (b,) * self.ndim

        assert len(b) == self.ndim, 'b must have the same dimensionality as the coords.'

        for i in range(len(self.coords)):
            self.coords[i] += b[i]

        return self

    def __isub__(self, b):
        '''Subtract `b` from the coordinates separately in-place
        '''
        if not isinstance(b, Iterable):
            b = (b,) * self.ndim

        assert len(b) == self.ndim, 'b must have the same dimensionality as the coords.'

        for i in range(len(self.coords)):
            self.coords[i] -= b[i]

        return self

    def __imul__(self, f):
        '''Multiply each coordinate with `f` separately in-place.
        '''
        if not isinstance(f, Iterable):
            f = (f,) * self.ndim

        assert len(f) == self.ndim, 'f must have the same dimensionality as the coords.'

        for i in range(len(self.coords)):
            self.coords[i] *= f[i]

        return self

    def __eq__(self, other):
        '''Check if the coordinates are identical.

        Parameters
        ----------
        other : object
            The object to which to compare.

        Returns
        -------
        boolean
            Whether the two objects are identical.
        '''
        if type(self) is not type(other):
            return False

        return all(self._xp.array_equal(s, o) for s, o in zip(self.coords, other.coords))

    def reverse(self):
        '''Reverse the ordering of points in-place.
        '''
        for i in range(len(self.coords)):
            self.coords[i] = self.coords[i][::-1]
        return self


class SeparatedCoords(Coords):
    '''A list of points that are separable along each dimension.

    The actual points are given by the iterated tensor product of the `separated_coords`.

    Parameters
    ----------
    separated_coords : list or tuple
        A tuple of a list of coordinates along each dimension.
    xp : module, optional
        The array namespace (e.g., numpy, cupy, jax.numpy). If not provided,
        it will be inferred from the input arrays.

    Attributes
    ----------
    separated_coords
        A tuple of a list of coordinates along each dimension.
    '''
    def __init__(self, separated_coords, xp=None):
        if xp is None:
            xp = _infer_xp(*separated_coords)
        self._xp = xp
        # Make a copy to avoid modification from outside the class
        self.separated_coords = [self._xp.asarray(s, dtype='float', copy=True) for s in separated_coords]

    @classmethod
    def from_dict(cls, tree):
        '''Make an SeparatedCoords from a dictionary, previously created by `to_dict()`.

        Parameters
        ----------
        tree : dictionary
            The dictionary from which to make a new SeparatedCoords object.

        Returns
        -------
        SeparatedCoords
            The created object.

        Raises
        ------
        ValueError
            If the dictionary is not formatted correctly.
        '''
        if tree['type'] != 'separated':
            raise ValueError('The type of coordinates should be "separated".')

        return cls(tree['separated_coords'])

    def to_dict(self):
        '''Convert the object to a dictionary for serialization.

        Returns
        -------
        dictionary
            The created dictionary.
        '''
        tree = {
            'type': 'separated',
            'separated_coords': self.separated_coords
        }

        return tree

    def __getstate__(self):
        '''Get state for pickling.
        '''
        return {
            'separated_coords': self.separated_coords,
            'xp_name': self._xp.__name__
        }

    def __setstate__(self, state):
        '''Restore state from pickle.
        '''
        self.separated_coords = state['separated_coords']
        # Restore xp from module name
        if state['xp_name'] == 'numpy':
            self._xp = np
        else:
            # For other backends, import dynamically
            import importlib
            self._xp = importlib.import_module(state['xp_name'])

    def __getitem__(self, i):
        '''The `i`-th point for these coordinates.
        '''
        s0 = (1,) * len(self)
        j = len(self) - i - 1
        output = self.separated_coords[i].reshape(s0[:j] + (-1,) + s0[j + 1:])
        return self._xp.broadcast_to(output, self.shape).ravel()

    @property
    def size(self):
        '''The number of points.
        '''
        return self._xp.prod(self._xp.asarray(self.shape))

    def __len__(self):
        '''The number of dimensions.
        '''
        return len(self.separated_coords)

    @property
    def dims(self):
        '''The number of points along each dimension.
        '''
        return tuple(len(c) for c in self.separated_coords)

    @property
    def shape(self):
        '''The shape of an ``numpy.ndarray`` with the right dimensions.
        '''
        return self.dims[::-1]

    def __iadd__(self, b):
        '''Add `b` to the coordinates separately in-place.
        '''
        if not isinstance(b, Iterable):
            b = (b,) * self.ndim

        assert len(b) == self.ndim, 'b must have same dimensionality as the coordinates.'

        for i in range(self.ndim):
            self.separated_coords[i] += b[i]

        return self

    def __isub__(self, b):
        '''Subtract `b` from the coordinates separately in-place.
        '''
        if not isinstance(b, Iterable):
            b = (b,) * self.ndim

        assert len(b) == self.ndim, 'b must have same dimensionality as the coordinates.'

        for i in range(self.ndim):
            self.separated_coords[i] -= b[i]

        return self

    def __imul__(self, f):
        '''Multiply each coordinate with `f` separately in-place.
        '''
        if not isinstance(f, Iterable):
            f = (f,) * self.ndim

        assert len(f) == self.ndim, 'f must have same dimensionality as the coordinates.'

        for i in range(self.ndim):
            self.separated_coords[i] *= f[i]

        return self

    def __eq__(self, other):
        '''Check if the coordinates are identical.

        Parameters
        ----------
        other : object
            The object to which to compare.

        Returns
        -------
        boolean
            Whether the two objects are identical.
        '''
        if type(self) is not type(other):
            return False

        if len(self.separated_coords) != len(other.separated_coords):
            return False

        for s, o in zip(self.separated_coords, other.separated_coords):
            if not self._xp.array_equal(s, o):
                return False

        return True

    def reverse(self):
        '''Reverse the ordering of points in-place.
        '''
        for i in range(len(self)):
            self.separated_coords[i] = self.separated_coords[i][::-1]
        return self


class RegularCoords(Coords):
    '''A list of points that have a regular spacing in all dimensions.

    Parameters
    ----------
    delta : array_like
        The spacing between the points.
    dims : tuple
        The number of points along each dimension.
    zero : array_like
        The coordinates for the first point.
    xp : module, optional
        The array namespace (e.g., numpy, cupy, jax.numpy). If not provided,
        it will be inferred from the delta and zero arrays.

    Attributes
    ----------
    delta
        The spacing between the points.
    dims
        The number of points along each dimension.
    zero
        The coordinates for the first point.
    '''
    def __init__(self, delta, dims, zero, xp=None):
        if not isinstance(dims, Iterable):
            raise ValueError('Dims should be an iterable.')

        # Try to infer xp from delta and zero if not provided
        if xp is None:
            try:
                xp = _infer_xp(delta, zero)
            except ValueError:
                # _infer_xp already raises ValueError for new-style fields
                raise

        self._xp = xp

        # Convert to arrays - accept tuples, lists, or arrays
        self.delta = xp.asarray(delta, dtype=float)
        self.dims = tuple(int(n) for n in dims)
        self.zero = xp.asarray(zero, dtype=float)

        # Validate dimensions match
        assert len(self.delta) == len(self.dims), 'delta and dims must have same length'
        assert len(self.dims) == len(self.zero), 'dims and zero must have same length'

    @classmethod
    def from_dict(cls, tree):
        '''Make an RegularCoords from a dictionary, previously created by `to_dict()`.

        Parameters
        ----------
        tree : dictionary
            The dictionary from which to make a new RegularCoords object.

        Returns
        -------
        RegularCoords
            The created object.

        Raises
        ------
        ValueError
            If the dictionary is not formatted correctly.
        '''
        if tree['type'] != 'regular':
            raise ValueError('The type of coordinates should be "regular".')

        # Restore xp from stored name
        if tree.get('xp_name') == 'numpy':
            xp = np
        elif 'xp_name' in tree:
            import importlib
            xp = importlib.import_module(tree['xp_name'])
        else:
            # Backward compatibility: default to numpy
            xp = np

        return cls(tree['delta'], tree['dims'], tree['zero'], xp=xp)

    def to_dict(self):
        '''Convert the object to a dictionary for serialization.

        Returns
        -------
        dictionary
            The created dictionary.
        '''
        # Convert arrays to lists for JSON serialization, store xp name
        tree = {
            'type': 'regular',
            'delta': self.delta.tolist(),
            'dims': self.dims,
            'zero': self.zero.tolist(),
            'xp_name': self._xp.__name__
        }

        return tree

    def __getstate__(self):
        '''Get state for pickling.
        '''
        # Convert arrays to lists for pickling
        return {
            'delta': self.delta.tolist(),
            'dims': self.dims,
            'zero': self.zero.tolist(),
            'xp_name': self._xp.__name__
        }

    def __setstate__(self, state):
        '''Restore state from pickle.
        '''
        self.dims = state['dims']
        # Restore xp from module name
        if state['xp_name'] == 'numpy':
            self._xp = np
        else:
            # For other backends, import dynamically
            import importlib
            self._xp = importlib.import_module(state['xp_name'])
        # Convert lists back to arrays
        self.delta = self._xp.asarray(state['delta'], dtype=float)
        self.zero = self._xp.asarray(state['zero'], dtype=float)

    @property
    def separated_coords(self):
        '''A tuple of a list of the values for each dimension.

        The actual points are the iterated tensor product of this tuple.
        '''
        return [self._xp.arange(n) * delta + zero for delta, n, zero in zip(self.delta, self.dims, self.zero)]

    @property
    def regular_coords(self):
        '''The tuple `(delta, dims, zero)` of the regularly-spaced coordinates.
        '''
        return self.delta, self.dims, self.zero

    @property
    def size(self):
        '''The number of points.
        '''
        return math.prod(self.dims)

    def __len__(self):
        '''The number of dimensions.
        '''
        return len(self.dims)

    @property
    def shape(self):
        '''The shape of an ``numpy.ndarray`` with the right dimensions.
        '''
        return tuple(reversed(self.dims))

    def __getitem__(self, i):
        '''The `i`-th point for these coordinates.
        '''
        s0 = (1,) * len(self)
        j = len(self) - i - 1
        t = s0[:j] + (-1,) + s0[j + 1:]
        output = self.separated_coords[i].reshape(t)
        return self._xp.broadcast_to(output, self.shape).ravel()

    def __iadd__(self, b):
        '''Add `b` to the coordinates separately in-place.
        '''
        if not isinstance(b, Iterable):
            b = (b,) * self.ndim

        assert len(b) == len(self), 'b must have same dimensionality as the coordinates.'

        self.zero = self.zero + self._xp.asarray(b, dtype=float)

        return self

    def __isub__(self, b):
        '''Subtract `b` from the coordinates separately in-place.
        '''
        if not isinstance(b, Iterable):
            b = (b,) * self.ndim

        assert len(b) == len(self), 'b must have same dimensionality as the coordinates.'

        self.zero = self.zero - self._xp.asarray(b, dtype=float)

        return self

    def __imul__(self, f):
        '''Multiply each coordinate with `f` separately in-place.
        '''
        if not isinstance(f, Iterable):
            f = (f,) * self.ndim

        assert len(f) == self.ndim, 'f must have same dimensionality as the coordinates.'

        f_arr = self._xp.asarray(f, dtype=float)
        self.delta = self.delta * f_arr
        self.zero = self.zero * f_arr

        return self

    def __eq__(self, other):
        '''Check if the coordinates are identical.

        Parameters
        ----------
        other : object
            The object to which to compare.

        Returns
        -------
        boolean
            Whether the two objects are identical.
        '''
        if type(self) is not type(other):
            return False

        # Use xp.all() for array comparison (Array API compatible)
        delta_equal = self._xp.all(self.delta == other.delta)
        zero_equal = self._xp.all(self.zero == other.zero)
        return delta_equal and self.dims == other.dims and zero_equal

    def reverse(self):
        '''Reverse the ordering of points in-place.
        '''
        self.zero = self.zero + self.delta * (self._xp.asarray(self.dims) - 1)
        self.delta = -self.delta

        return self

Coords._add_coordinate_type('unstructured', UnstructuredCoords)
Coords._add_coordinate_type('separated', SeparatedCoords)
Coords._add_coordinate_type('regular', RegularCoords)
