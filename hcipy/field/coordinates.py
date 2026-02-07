import numpy as np
import copy
import math
from collections.abc import Iterable
import array_api_compat
from ..config import Configuration

def _infer_xp(arrays):
    """Infer xp from input arrays.
    
    Parameters
    ----------
    arrays : list or tuple
        Input arrays to infer xp from
        
    Returns
    -------
    xp : module
        The inferred xp module, or numpy as fallback
    """
    if arrays is not None and len(arrays) > 0:
        try:
            return array_api_compat.array_namespace(*arrays)
        except (ValueError, TypeError):
            pass
    return np


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
        A tuple of a list of positions for each dimension. The backend is
        inferred from the input arrays.
    '''
    def __init__(self, coords):
        self._xp = _infer_xp(coords)
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
        A tuple of a list of coordinates along each dimension. The backend is
        inferred from the input arrays.

    Attributes
    ----------
    separated_coords
        A tuple of a list of coordinates along each dimension.
    '''
    def __init__(self, separated_coords):
        self._xp = _infer_xp(separated_coords)
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
    dims : array_like
        The number of points along each dimension.
    zero : array_like
        The coordinates for the first point.
    xp : module, optional
        The array namespace (e.g., numpy, cupy, jax.numpy). If not provided,
        it will be inferred from the input arrays. If provided and arrays have
        a different backend, an error will be raised. For RegularCoords, since
        the parameters are scalars, xp must be explicitly provided if 
        use_new_style_fields is True.

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
        if not isinstance(delta, Iterable):
            raise ValueError('Delta should be an iterable.')

        if not isinstance(dims, Iterable):
            raise ValueError('Dims should be an iterable.')

        if not isinstance(zero, Iterable):
            raise ValueError('Zero should be an iterable.')

        delta = tuple(float(d) for d in delta)
        dims = tuple(int(n) for n in dims)
        zero = tuple(float(z) for z in zero)

        assert len(delta) == len(dims)
        assert len(dims) == len(zero)

        # For RegularCoords, we can't infer xp from scalar parameters
        # So xp must be provided explicitly if use_new_style_fields is True
        if xp is None:
            if Configuration().core.use_new_style_fields:
                raise ValueError("xp must be specified for RegularCoords with new-style Fields")
            xp = np
        
        self._xp = xp
        self.dims = dims
        self.delta = delta
        self.zero = zero

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

        return cls(tree['delta'], tree['dims'], tree['zero'])

    def to_dict(self):
        '''Convert the object to a dictionary for serialization.

        Returns
        -------
        dictionary
            The created dictionary.
        '''
        tree = {
            'type': 'regular',
            'delta': self.delta,
            'dims': self.dims,
            'zero': self.zero
        }

        return tree

    def __getstate__(self):
        '''Get state for pickling.
        '''
        return {
            'delta': self.delta,
            'dims': self.dims,
            'zero': self.zero,
            'xp_name': self._xp.__name__
        }

    def __setstate__(self, state):
        '''Restore state from pickle.
        '''
        self.delta = state['delta']
        self.dims = state['dims']
        self.zero = state['zero']
        # Restore xp from module name
        if state['xp_name'] == 'numpy':
            self._xp = np
        else:
            # For other backends, import dynamically
            import importlib
            self._xp = importlib.import_module(state['xp_name'])

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

        self.zero = tuple(aa + bb for aa, bb in zip(self.zero, b))

        return self

    def __isub__(self, b):
        '''Subtract `b` from the coordinates separately in-place.
        '''
        if not isinstance(b, Iterable):
            b = (b,) * self.ndim

        assert len(b) == len(self), 'b must have same dimensionality as the coordinates.'

        self.zero = tuple(aa - bb for aa, bb in zip(self.zero, b))

        return self

    def __imul__(self, f):
        '''Multiply each coordinate with `f` separately in-place.
        '''
        if not isinstance(f, Iterable):
            f = (f,) * self.ndim

        assert len(f) == self.ndim, 'f must have same dimensionality as the coordinates.'

        self.delta = tuple(d * ff for d, ff in zip(self.delta, f))
        self.zero = tuple(z * ff for z, ff in zip(self.zero, f))

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

        return self.delta == other.delta and self.dims == other.dims and self.zero == other.zero

    def reverse(self):
        '''Reverse the ordering of points in-place.
        '''
        self.zero = tuple(z + d * (n - 1) for z, d, n in zip(self.zero, self.delta, self.dims))
        self.delta = tuple(-d for d in self.delta)

        return self

Coords._add_coordinate_type('unstructured', UnstructuredCoords)
Coords._add_coordinate_type('separated', SeparatedCoords)
Coords._add_coordinate_type('regular', RegularCoords)
