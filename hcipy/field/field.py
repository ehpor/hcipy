import numpy as onp
import jax.numpy as jnp
from ..config import Configuration
from ..math import numpy as np

class Field:
    '''The value of some physical quantity for each point in some coordinate system.

    Parameters
    ----------
    arr : array_like
        An array of values or tensors for each point in the :class:`Grid`.
    grid : Grid
        The corresponding :class:`Grid` on which the values are set.

    Attributes
    ----------
    grid : Grid
        The grid on which the values are defined.
    '''
    def __new__(cls, arr, grid):
        if Configuration().core.use_new_style_fields:
            return super().__new__(NewStyleField)
        else:
            return OldStyleField.__new__(OldStyleField, arr, grid)

    @classmethod
    def from_dict(cls, tree):
        '''Make a Field from a dictionary, previously created by `to_dict()`.

        Parameters
        ----------
        tree : dictionary
            The dictionary from which to make a new Field object.

        Returns
        -------
        Field
            The created object.

        Raises
        ------
        ValueError
            If the dictionary is not formatted correctly.
        '''
        from .grid import Grid

        return cls(np.array(tree['values']), Grid.from_dict(tree['grid']))

    def to_dict(self):
        '''Convert the object to a dictionary for serialization.

        Returns
        -------
        dictionary
            The created dictionary.
        '''
        tree = {
            'values': np.asarray(self),
            'grid': self.grid.to_dict()
        }

        return tree

    def __reduce__(self):
        '''Return a 3-tuple for pickling the Field.

        Returns
        -------
        tuple
            The reduced version of the Field.
        '''
        return (
            _field_reconstruct,
            (self.__class__, np.ndarray, (0,), 'b',),
            self.__getstate__()
        )

    @property
    def tensor_order(self):
        '''The order of the tensor of the field.
        '''
        return self.ndim - 1

    @property
    def tensor_shape(self):
        '''The shape of the tensor of the field.
        '''
        return np.array(self.shape)[:-1]

    @property
    def is_scalar_field(self):
        '''True if this field is a scalar field (ie. a tensor order of 0), False otherwise.
        '''
        return self.tensor_order == 0

    @property
    def is_vector_field(self):
        '''True if this field is a vector field (ie. a tensor order of 1), False otherwise.
        '''
        return self.tensor_order == 1

    @property
    def is_valid_field(self):
        '''True if the field corresponds with its grid.
        '''
        return self.shape[-1] == self.grid.size

    @property
    def shaped(self):
        '''The reshaped version of this field.

        Raises
        ------
        ValueError
            If this field isn't separated, no reshaped version can be made.
        '''
        if not self.grid.is_separated:
            raise ValueError('This field doesn\'t have a shape.')

        if self.tensor_order > 0:
            new_shape = np.concatenate([np.array(self.shape)[:-1], self.grid.shape])
            return self.reshape(new_shape)

        return self.reshape(self.grid.shape)

    def at(self, p):
        '''The value of this field closest to point p.

        Parameters
        ----------
        p : array_like
            The point at which the closest value should be returned.

        Returns
        -------
        array_like
            The value, potentially tensor, closest to point p.
        '''
        i = self.grid.closest_to(p)
        return self[..., i]

class OldStyleField(Field, np.ndarray):
    '''A Field based on subclassing a Numpy array.

    This constitutes an "old-style" Field object. Due to problems and inflexibilities
    in the way subclassing works, we are gradually transitioning to "new-style" fields
    (see :class:`NewStyleField`) which use the dispatch mechanism introduced by
    Numpy 1.13.

    Parameters
    ----------
    arr : array_like
        An array of values or tensors for each point in the :class:`Grid`.
    grid : Grid
        The corresponding :class:`Grid` on which the values are set.

    Attributes
    ----------
    grid : Grid
        The grid on which the values are defined.
    '''
    def __new__(cls, arr, grid):
        obj = np.asarray(arr).view(cls)
        obj.grid = grid
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.grid = getattr(obj, 'grid', None)

    def __getstate__(self):
        '''Get the internal state for pickling.

        Returns
        -------
        tuple
            The state of the Field.
        '''
        data_state = np.ndarray.__reduce__(self)[2]
        return data_state + (self.grid,)

    def __setstate__(self, state):
        '''Set the internal state for pickling.

        Parameters
        ----------
        state : tuple
            The state coming from a __getstate__().
        '''
        _, shp, typ, isf, raw, grid = state

        super().__setstate__((shp, typ, isf, raw))
        self.grid = grid

def _field_reconstruct(subtype, baseclass, baseshape, basetype):
    '''Internal function for building a new Field object for pickling.

    Parameters
    ----------
    subtype
        The class of Field.
    baseclass
        The array class that was used for the Field.
    baseshape
        The shape of the Field.
    basetype
        The data type of the Field.

    Returns
    -------
    Field
        The built Field object.
    '''
    data = np.ndarray.__new__(baseclass, baseshape, basetype)
    grid = None

    return subtype.__new__(subtype, data, grid)

def _unwrap(arg):
    if isinstance(arg, Field):
        return arg.data

    if isinstance(arg, list):
        return [_unwrap(x) for x in arg]

    if isinstance(arg, dict):
        return {key: _unwrap(val) for key, val in arg.items()}

    if isinstance(arg, tuple):
        return tuple(_unwrap(x) for x in arg)

    return arg

class NewStyleField(Field, onp.lib.mixins.NDArrayOperatorsMixin): #TODO This should probably not refer back to original numpy
    '''A Field based on composition rather than subclassing a Numpy array.

    This constitutes an "new-style" Field object. A previous version uses subclassing
    from Numpy arrays (see :class:`OldStyleField` that had problems and inflexibilities
    due to the way subclassing works. We are gradually transitioning to "new-style" fields,
    which use the dispatch mechanism introduced by Numpy 1.13.

    Parameters
    ----------
    data : array_like
        An array of values or tensors for each point in the :class:`Grid`.
    grid : Grid
        The corresponding :class:`Grid` on which the values are set.

    Attributes
    ----------
    data : array_like
        The underlying data as a Numpy array or compatible object.
    grid : Grid
        The grid on which the values are defined.
    '''
    def __init__(self, data, grid):
        self.data = np.asarray(data)
        self.grid = grid

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        out = kwargs.get('out', ())

        inputs = _unwrap(inputs)

        if out:
            kwargs['out'] = tuple(x.data if is_field(x) else x for x in out)

        if hasattr(jnp, ufunc.__name__):
            # make sure jax (or numpy) ufuncs are dispatched to internal np
            result = getattr(np, ufunc.__name__)(*inputs, **kwargs) 
        else:
            result = getattr(ufunc, method)(*inputs, **kwargs)

        # Note: this is an extremely simple way of determining whether the
        # resulting object should be a Field or not. We can likely do better
        # by looking directly at the name of the function that we are executing.
        # The current code is fine for most ufuncs.
        if isinstance(result, np.ndarray) and not np.isscalar(result):
            return Field(result, self.grid)
        else:
            return result

    def __array__(self, dtype=None, copy=None):
        if dtype is None or dtype == self.data.dtype:
            # A no-copy is supported if necessary.
            if copy:
                return self.data.copy()
            else:
                return self.data

        if copy is False:
            raise ValueError(f'copy=False is not supported since the dtypes do not match ({dtype} vs. {self.data.dtype}).')

        return self.data.astype(dtype)
    
    def __jax_array__(self, dtype=None, copy=None):
        return self.__array__(dtype, copy)

    def __array_function__(self, func, types, args, kwargs):
        args = _unwrap(args)

        result = func(*args, **kwargs)

        # Note: this is an extremely simple way of determining whether the
        # resulting object should be a Field or not. We can likely do better
        # by looking directly at the name of the function that we are executing.
        if isinstance(result, np.ndarray) and not np.isscalar(result):
            return Field(result, self.grid)

        return result

    def tree_flatten(self):
        return (self.data,), self.grid
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(children[0], aux_data)

    def __getitem__(self, indices):
        res = self.data[indices]
        if np.isscalar(res):
            return res

        return Field(res, self.grid)

    def __setitem__(self, indices, values):
        self.data[indices] = values

        return self

    def __getstate__(self):
        '''Get the internal state for pickling.

        Returns
        -------
        tuple
            The state of the Field.
        '''
        #TODO make this also compatible with JAX arrays
        if isinstance(self.data, jnp.ndarray):
            data_state = self.data.__reduce__()[1][2]
        else:
            data_state = self.data.__reduce__()[2]
        return data_state + (self.grid,)

    def __setstate__(self, state):
        '''Set the internal state for pickling.

        Parameters
        ----------
        state : tuple
            The state coming from a __getstate__().
        '''
        _, shp, typ, isf, raw, grid = state

        self.data = np.array([])
        self.data.__setstate__((shp, typ, isf, raw))
        self.grid = grid
    
    def matvec(self, x):
        return np.asarray(self.data) @ x
    
    def rmatvec(self, x):
        return x @ np.asarray(self.data)
    
    @property
    def T(self):
        return np.transpose(self)

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def imag(self):
        return np.imag(self)

    @property
    def real(self):
        return np.real(self)

    @property
    def size(self):
        return np.prod(np.asarray(self.shape))

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def shape(self):
        return self.data.shape

    def astype(self, dtype, *args, **kwargs):
        return Field(self.data.astype(dtype, *args, **kwargs), self.grid)

    def __len__(self):
        return len(self.data)

    def conj(self, *args, **kwargs):
        return np.conj(self, *args, **kwargs)

    def conjugate(self, *args, **kwargs):
        return np.conjugate(self, *args, **kwargs)

    #TODO The following additions are because of the arguments not being ordered correctly when using the regular dispatcher or missing an argument (e.g. array.sum() where it misses self). Either fix this or remove the commented out lines in the definitions below. It probably also happens for the remaining functions but they aren't covered in the tests
    def clip(self, *args, **kwargs):
        return np.clip(self, *args, **kwargs)

    def repeat(self, *args, **kwargs):
        return np.repeat(self, *args, **kwargs)

    def dot(self, *args, **kwargs):
        return np.dot(self, *args, **kwargs)

    def reshape(self, *args, **kwargs):
        return np.reshape(self, *args, **kwargs)
    
    def sum(self, *args, **kwargs):
        return np.sum(self, *args, **kwargs)
    
    def mean(self, *args, **kwargs):
        return np.mean(self, *args, **kwargs)

    def std(self, *args, **kwargs):
        return np.std(self, *args, **kwargs)
    
    def max(self, *args, **kwargs):
        return np.max(self, *args, **kwargs)
    
    def min(self, *args, **kwargs):
        return np.min(self, *args, **kwargs)
    
    def ravel(self, *args, **kwargs):
        return np.ravel(self, *args, **kwargs)

    def copy(self, *args, **kwargs):
        return np.copy(self, *args, **kwargs)
    
    def round(self, *args, **kwargs):
        return np.round(self, *args, **kwargs)
    

    @property
    def flags(self):
        return self.data.flags

    all = np.all
    any = np.any
    argmax = np.argmax
    argmin = np.argmin
    argpartition = np.argpartition
    argsort = np.argsort
    # clip = np.clip
    compress = np.compress
    # copy = np.copy
    cumprod = np.cumprod
    cumsum = np.cumsum
    # dot = np.dot
    flatten = np.ravel
    # max = np.max
    # mean = np.mean
    # min = np.min
    nonzero = np.nonzero
    prod = np.prod
    # ravel = np.ravel
    # repeat = np.repeat
    # reshape = np.reshape
    # round = np.round
    sort = np.sort
    squeeze = np.squeeze
    # std = np.std
    # sum = np.sum
    trace = np.trace
    transpose = np.transpose
    var = np.var

def is_field(obj):
    '''Check if the object is an HCIPy Field.

    This function should be preferred over of `isinstance(obj, hcipy.Field)` for
    forward compatibility reasons.

    Parameters
    ----------
    obj : Any
        Any object to check.

    Returns
    -------
    boolean
        Whether `obj` is an HCIPy Field or not.
    '''
    return isinstance(obj, Field)
