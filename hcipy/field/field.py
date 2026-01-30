import numpy as np
from ..config import Configuration

class FieldBase:
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
    @property
    def tensor_order(self):
        '''The order of the tensor of the field.
        '''
        return self.ndim - 1

    @property
    def tensor_shape(self):
        '''The shape of the tensor of the field.
        '''
        return self.shape[:-1]

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

        new_shape = self.shape[:-1] + self.grid.shape
        return self.reshape(new_shape)

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

class OldStyleField(FieldBase, np.ndarray):
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

def _make_binary_operator(Field, op):
    src = f"""def func(self, other, /):
        if isinstance(other, Field):
            return Field(self.data {op} other.data, self.grid)
        return Field(self.data {op} other, self.grid)"""

    ns = {'Field': Field}
    exec(src, ns)

    ns['func'].__doc__ = f"Elementwise operator {op}."

    return ns['func']

def _make_inplace_binary_operator(Field, op):
    src = f"""def func(self, other, /):
        if isinstance(other, Field):
            self.data {op}= other.data
        else:
            self.data {op}= other
        return self"""

    ns = {'Field': Field}
    exec(src, ns)

    ns['func'].__doc__ = f"Elementwise inplace operator {op}."

    return ns['func']

def _make_reflected_binary_operator(Field, op):
    src = f"""def func(self, other, /):
        if isinstance(other, Field):
            return Field(other.data {op} self.data, self.grid)
        return Field(other {op} self.data, self.grid)"""

    ns = {'Field': Field}
    exec(src, ns)

    ns['func'].__doc__ = f"Elementwise reflected operator {op}."

    return ns['func']

def _make_array_api_func(func_name, sig):
    args = [s.strip() for s in sig.split(',')]

    call_args = []

    for i, arg in enumerate(args):
        if arg in ['/', '*']:
            continue

        if '=' in arg:
            # Keyword argument.
            name = arg.split('=')[0]
            call_args.append(f'{name}={name}')
        else:
            # Positional argument.
            call_args.append(arg)

    src = f"""def func({', '.join(args)}):
        xp = {args[0]}.__array_namespace__()
        return xp.{func_name}({', '.join(call_args)})
        """

    ns = {}
    exec(src, ns)

    return ns['func']

def _unwrap(arg):
    if type(arg) is NewStyleField:
        return arg.data

    if type(arg) is tuple:
        # Fast paths for 2 and 3 length tuples as those are most common.
        length = len(arg)

        if length == 2:
            a, b = arg
            return _unwrap(a), _unwrap(b)

        if length == 3:
            a, b, c = arg
            return _unwrap(a), _unwrap(b), _unwrap(c)

        return tuple([_unwrap(x) for x in arg])

    if type(arg) is list:
        return [_unwrap(x) for x in arg]

    if type(arg) is dict:
        return {key: _unwrap(val) for key, val in arg.items()}

    return arg

class NewStyleField(FieldBase):
    '''The value of some physical quantity for each point in some coordinate system.

    Parameters
    ----------
    data : array or Field
        An array of values or tensors for each point in the :class:`Grid`.
    grid : Grid or None
        The corresponding :class:`Grid` on which these values are set. The default (None)
        indicates that there is no grid associated with this Field.

    Attributes
    ----------
    data : array
        An array of values or tensors for each point in the :class:`Grid`.
    grid : Grid or None
        The corresponding :class:`Grid` on which these values are set. When this is None,
        this indicates that there is no grid associated with this Field.
    '''
    __slots__ = ('data', 'grid')

    def __init__(self, data, grid=None):
        self.data = data.data if isinstance(data, NewStyleField) else data
        self.grid = grid

    def __abs__(self):
        return NewStyleField(abs(self.data), self.grid)

    def __pos__(self):
        return NewStyleField(+self.data, self.grid)

    def __neg__(self):
        return NewStyleField(-self.data, self.grid)

    def __invert__(self):
        return NewStyleField(~self.data, self.grid)

    def __bool__(self):
        return self.data.__bool__()

    def __complex__(self):
        return complex(self.data)

    def __float__(self):
        return float(self.data)

    def __int__(self):
        return int(self.data)

    def __index__(self):
        return self.data.__index__()

    def __getitem__(self, key, /):
        xp = self.data.__array_namespace__()
        return NewStyleField(xp.asarray(self.data[key]), self.grid)

    def __setitem__(self, key, value, /):
        self.data[key] = value

        return self

    @property
    def dtype(self):
        '''The data type of the array elements.'''
        return self.data.dtype

    @property
    def device(self):
        '''The hardware device the array data resides on.'''
        return self.data.device

    @property
    def mT(self):
        '''The transpose of a matrix or stack of matrices.'''
        return NewStyleField(self.data.mT, self.grid)

    @property
    def ndim(self):
        '''The number of axes in the array.'''
        return self.data.ndim

    @property
    def shape(self):
        '''The dimensions of the array.'''
        return self.data.shape

    @property
    def size(self):
        '''The number of elements in the array.'''
        return self.data.size

    @property
    def T(self):
        '''Tranpose of the array.'''
        return NewStyleField(self.data.T, self.grid)

    def __dlpack__(self, *, stream=None, max_version=None, dl_device=None, copy=None):
        '''Exports the Field for consumption by from_dlpack() as a DLPack capsule.

        Parameters
        ----------
        stream : stream or None
            The stream to use. For CUDA and ROCm, a Python integer representing a pointer to a stream,
            on devices that support streams. `stream` is provided by the consumer to the producer to
            instruct the producer to ensure that operations can safely be performed on the array (e.g.,
            by inserting a dependency between streams via “wait for event”). The pointer must be an
            integer larger than or equal to -1 (see below for allowed values on each platform). If
            stream is -1, the value may be used by the consumer to signal “producer must not perform
            any synchronization”. The ownership of the stream stays with the consumer. On CPU and
            other device types without streams, only None is accepted.
        max_version : tuple of ints or None
            the maximum DLPack version that the consumer (i.e., the caller of `__dlpack__`) supports,
            in the form of a 2-tuple (major, minor). This method may return a capsule of version
            `max_version`, or of a different version. This means the consumer must verify the version
            even when `max_version` is passed.
        dl_device : tuple or None
            the DLPack device type. Default is None, meaning the exported capsule should be on the same
            device as self is. When specified, the format must be a 2-tuple, following that of the
            return value of array.__dlpack_device__(). If the device type cannot be handled by the
            producer, this function must raise BufferError.
        copy : bool or None
            Whether or not to copy the input. If True, the function must always copy. If False, the
            function must never copy, and raise a BufferError in case a copy is deemed necessary
            (e.g. if a cross-device data movement is requested, and it is not possible without a copy).
            If None, the function must reuse the existing memory buffer if possible and copy otherwise.
            Default: None.

        Returns
        -------
        capsule
            A DLPack capsule for the Field.

        Raises
        ------
        BufferError
            When the data cannot be exported as DLPack (e.g., incompatible dtype or strides). Other errors
            are raised when export fails for other reasons (e.g., incorrect arguments passed or out of memory).
        '''
        return self.data.__dlpack__(stream=stream, max_version=max_version, dl_device=dl_device, copy=copy)

    def __dlpack_device__(self):
        '''Returns device type and device ID in DLPack format. Meant for use within `from_dlpack()`.

        Returns
        -------
        tuple
            The device type and device ID in DLPack format.
        '''
        return self.data.__dlpack_device__()

    def to_device(self, device, /, *, stream=None):
        '''Copy the Field from the device on which it currently resides to the specified device.

        Parameters
        ----------
        device : device
            A device object.
        stream : stream or None
            The stream to use. For CUDA and ROCm, a Python integer representing a pointer to a stream,
            on devices that support streams. `stream` is provided by the consumer to the producer to
            instruct the producer to ensure that operations can safely be performed on the array (e.g.,
            by inserting a dependency between streams via “wait for event”). The pointer must be an
            integer larger than or equal to -1 (see below for allowed values on each platform). If
            stream is -1, the value may be used by the consumer to signal “producer must not perform
            any synchronization”. The ownership of the stream stays with the consumer. On CPU and
            other device types without streams, only None is accepted.

        Returns
        -------
        Field
            An array with the same data and data type as self and located on the specified device.
        '''
        return NewStyleField(self.data.to_device(device, stream=stream), self.grid)

    def __array_namespace__(self, api_version=None):
        '''Returns an object that has all the array API functions on it.

        Parameters
        ----------
        api_version : str or None
            A string representing the version of the array API specification to be returned. If this
            is None, it will return the namespace corresponding to the latest version of the array API
            specification. If the given version is invalid or not implemented for the given backend, an
            error will be raised. Default: None.

        Returns
        -------
        FieldNamespace
            An object representing the array API namespace.
        '''
        return make_field_namespace(self.data.__array_namespace__(api_version=api_version))

    def __getstate__(self):
        '''Return pickle state.

        Returns
        -------
        dict
            The state of the Field.

        Notes
        -----
        This method is used internally by pickle. The returned state must be
        fully pickleable. Array objects are serialized using their native
        pickle support (NumPy, CuPy, JAX).
        '''
        return {
            'data': self.data,
            'grid': self.grid,
        }

    def __setstate__(self, state):
        '''Restore object state from pickle.

        Parameters
        ----------
        state : dict
            State produced by ``__getstate__``.
        '''
        self.data = state["data"]
        self.grid = state["grid"]

    def __repr__(self):
        return f"Field(data={self.data}, grid={self.grid})"

    all = _make_array_api_func('all', 'self, /, *, axis=None, keepdims=False')
    any = _make_array_api_func('any', 'self, /, *, axis=None, keepdims=False')
    argmax = _make_array_api_func('argmax', 'self, /, *, axis=None, keepdims=False')
    argmin = _make_array_api_func('argmin', 'self, /, *, axis=None, keepdims=False')
    # argsort = _make_array_api_func('argsort', 'self, /, *, axis=-1, descending=False, stable=True')
    argsort = _make_array_api_func('argsort', 'self, /, *, axis=-1, stable=True')
    astype = _make_array_api_func('astype', 'self, dtype, /, *, copy=True, device=None')
    clip = _make_array_api_func('clip', 'self, /, min=None, max=None')
    cumprod = _make_array_api_func('cumulative_prod', 'self, /, *, axis=None, dtype=None, include_initial=False')
    cumulative_prod = cumprod
    cumsum = _make_array_api_func('cumulative_sum', 'self, /, *, axis=None, dtype=None, include_initial=False')
    cumulative_sum = cumsum
    max = _make_array_api_func('max', 'self, /, *, axis=None, keepdims=False')
    mean = _make_array_api_func('mean', 'self, /, *, axis=None, keepdims=False')
    min = _make_array_api_func('min', 'self, /, *, axis=None, keepdims=False')
    nonzero = _make_array_api_func('nonzero', 'self, /')
    prod = _make_array_api_func('prod', 'self, /, *, axis=None, dtype=None, keepdims=False')
    round = _make_array_api_func('round', 'self, /')
    reshape = _make_array_api_func('reshape', 'self, /, shape, *, copy=None')
    # sort = _make_array_api_func('sort', 'self, /, *, axis=-1, descending=False, stable=True')
    sort = _make_array_api_func('sort', 'self, /, *, axis=-1, stable=True')
    squeeze = _make_array_api_func('squeeze', 'self, /, *, axis=None')
    std = _make_array_api_func('std', 'self, /, *, axis=None, correction=0.0, keepdim=False')
    sum = _make_array_api_func('sum', 'self, /, *, axis=None, dtype=None, keepdims=False')
    var = _make_array_api_func('var', 'self, /, *, axis=None, correction=0.0, keepdim=False')

    @property
    def real(self):
        xp = self.__array_namespace__()
        return NewStyleField(xp.real(self.data), self.grid)

    @property
    def imag(self):
        xp = self.__array_namespace__()
        return NewStyleField(xp.imag(self.data), self.grid)

    def repeat(self, repeats, /, *, axis=None):
        xp = self.__array_namespace__()
        res = xp.repeat(self.data, repeats, axis=axis)

        return NewStyleField(res, self.grid)

    conj = _make_array_api_func('conj', 'self, /')
    conjugate = conj

    def copy(self, order='C'):
        xp = self.__array_namespace__()
        return xp.asarray(self, copy=True)

    def ravel(self):
        return self.reshape((-1,))

    flatten = ravel

    def dot(self, b):
        xp = self.__array_namespace__()

        a_ndim = self.ndim
        b_ndim = b.ndim

        # 2D · 2D → matrix multiply
        if a_ndim == 2 and b_ndim == 2:
            return xp.matmul(self, b)

        # 1D · 1D OR N-D · 1D
        if b_ndim == 1:
            return xp.asarray(xp.vecdot(self, b))

        # 1D · N-D (N >= 2)
        if a_ndim == 1:
            return xp.vecdot(self[..., None, :], b, axis=-2)

        # General N-D · M-D
        return xp.vecdot(self[..., None, :], b, axis=-2)

    def to_dict(self):
        '''Convert the Field to a dict.

        Returns
        -------
        dict
            The created dict.
        '''
        return {
            "values": self.__array__(),
            "grid": self.grid.to_dict()
        }

    @classmethod
    def from_dict(cls, val):
        '''Create a Field from a dict.

        Parameters
        ----------
        val : dict
            A dictionary generated from Field.to_dict().

        Returns
        -------
        Field
            The created field.
        '''
        from .grid import Grid

        return cls(val['values'], Grid.from_dict(val['grid']))

    trace = None
    transpose = None

    def __array__(self, dtype=None, copy=None):
        raise NotImplementedError("This Field cannot be used directly with Numpy. Use the Array API namespace instead.")

_aritmetic_and_bitwise_operators = [
    # Arithmetic
    ('add', '+'),
    ('sub', '-'),
    ('mul', '*'),
    ('truediv', '/'),
    ('floordiv', '//'),
    ('mod', '%'),
    ('pow', '**'),
    ('matmul', '@'),

    # Bitwise
    ('and', '&'),
    ('or', '|'),
    ('xor', '^'),
    ('lshift', '<<'),
    ('rshift', '>>'),
]

_comparison_operators = [
    ('lt', '<'),
    ('le', '<='),
    ('gt', '>'),
    ('ge', '>='),
    ('eq', '=='),
    ('ne', '!=')
]

for name, symbol in _aritmetic_and_bitwise_operators:
    setattr(NewStyleField, f'__{name}__', _make_binary_operator(NewStyleField, symbol))
    setattr(NewStyleField, f'__i{name}__', _make_inplace_binary_operator(NewStyleField, symbol))
    setattr(NewStyleField, f'__r{name}__', _make_reflected_binary_operator(NewStyleField, symbol))

for name, symbol in _comparison_operators:
    setattr(NewStyleField, f'__{name}__', _make_binary_operator(NewStyleField, symbol))

def _make_array_api_namespace_func(func, args, num_array_args=0):
    args = [arg.strip() for arg in args.split(',')]

    call_args = []

    for i, arg in enumerate(args):
        if arg in ['/', '*']:
            continue

        if '=' in arg:
            # Keyword argument.
            name = arg.split('=')[0]
            call_args.append(f'{name}={name}')
        else:
            # Positional argument.
            if i < num_array_args:
                # Inline xp._unwrap() for performance reasons.
                call_args.append(f'{arg}.data if isinstance({arg}, Field) else {arg}')
            else:
                call_args.append(arg)

    grid_src = ''.join(f'{arg}.grid if isinstance({arg}, Field) else ' for arg in args[:num_array_args])
    grid_src += 'None'

    src = f"""def {func.__name__}({', '.join(args)}):
        data = func({', '.join(call_args)})
        grid = {grid_src}
        return Field(data, grid)
        """

    ns = {'func': func, 'Field': NewStyleField}
    exec(src, ns)

    ns[func.__name__].__doc__ = func.__doc__

    return ns[func.__name__]

UNARY_OPS = [
    "abs", "acos", "acosh", "asin", "asinh", "atan", "atan2", "atanh",
    "bitwise_invert", "ceil", "conj", "cos", "cosh", "exp", "expm1",
    "floor", "imag", "isfinite", "isinf", "isnan", "log", "log1p", "log2", "log10",
    "logical_not", "negative", "positive", "real", "reciprocal", "round",
    "sign", "signbit", "sin", "sinh", "square", "sqrt", "tan", "tanh", "trunc",
]

BINARY_OPS = [
    "add", "bitwise_and", "bitwise_left_shift", "bitwise_or", "bitwise_right_shift", "bitwise_xor",
    "copysign", "divide", "equal", "floor_divide", "greater", "greater_equal", "hypot",
    "less", "less_equal", "logaddexp", "logical_and", "logical_or", "logical_xor",
    "maximum", "minimum", "multiply", "nextafter", "not_equal", "pow",
    "remainder", "subtract",
]

CONSTANTS = [
    "pi", "e", "inf", "nan", "newaxis"
]

FEEDTHROUGH_FUNCS = [
    "__array_namespace_info__",
    "can_cast", "finfo", "iinfo", "result_type",
    "bool",
    "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64",
    "float32", "float64", "complex64", "complex128",
]

OTHER_FUNCS = [
    "meshgrid",
    "broadcast_arrays",
    "concat",
    "stack",
]

ZERO_ARG_FUNCS = {
    "arange": "start, /, stop=None, step=1, *, dtype=None, device=None",
    "empty": "shape, *, dtype=None, device=None",
    "eye": "n_rows, n_cols=None, /, *, k=0, dtype=None, device=None",
    "from_dlpack": "x, /, *, device=None, copy=None",
    "full": "shape, fill_value, *, dtype=None",
    "linspace": "start, stop, /, num, *, dtype=None",
    "ones": "shape, *, dtype=None, device=None",
    "zeros": "shape, *, dtype=None, device=None",
}

ONE_ARG_FUNCS = {
    "all": "x, /, *, axis=None, keepdims=False",
    "any": "x, /, *, axis=None, keepdims=False",
    "argmax": "x, /, *, axis=None, keepdims=False",
    "argmin": "x, /, *, axis=None, keepdims=False",
    "argsort": "x, /, *, axis=-1, descending=False, stable=True",
    "asarray": "obj, /, *, dtype=None, device=None, copy=None",
    "astype": "x, dtype, /, *, copy=True, device=None",
    "broadcast_to": "x, dtype, /, *, copy=True, device=None",
    "clip": "x, /, min=None, max=None",
    "count_nonzero": "x, /, *, axis=None, keepdims=False",
    "cumulative_prod": "x, /, *, axis=None, dtype=None",
    "cumulative_sum": "x, /, *, axis=None, dtype=None",
    "diff": "x, /, *, axis=-1, prepend=None, append=None",
    "empty_like": "x, /, *, dtype=None, device=None",
    "expand_dims": "x, /, *, axis=0",
    "flip": "x, /, *, axis=None",
    "full_like": "x, /, fill_value, *, dtype=None, device=None",
    "matrix_transpose": "x, /",
    "max": "x, /, *, axis=None, keepdims=False",
    "mean": "x, /, *, axis=None, keepdims=False",
    "min": "x, /, *, axis=None, keepdims=False",
    "moveaxis": "x, source, destination, /",
    "nonzero": "x",
    "ones_like": "x, /, *, dtype=None, device=None",
    "permute_dims": "x, /, axes",
    "prod": "x, /, *, axis=None, dtype=None, keepdims=False",
    "repeat": "x, repeats, /, *, axis=None",
    "reshape": "x, /, shape, copy=None",
    "roll": "x, /, shift, *, axis=None",
    # "sort": "x, /, *, axis=-1, descending=False, stable=True",
    "sort": "x, /, *, axis=-1, stable=True",  # Numpy doesn't support the descending keyword yet.
    "squeeze": "x, /, axis",
    "std": "x, /, *, axis=None, correction=0.0, keepdims=False",
    "sum": "x, /, *, axis=None, dtype=None, keepdims=False",
    "tile": "x, repetitions, /",
    "tril": "x, /, *, k=0",
    "triu": "x, /, *, k=0",
    "unique_all": "x, /",
    "unique_counts": "x, /",
    "unique_inverse": "x, /",
    "unique_values": "x, /",
    "unstack": "x, /, *, axis=0",
    "var": "x, /, *, axis=None, correction=0.0, keepdims=False",
    "zeros_like": "x, /, *, dtype=None, device=None",
}

TWO_ARG_FUNCS = {
    "matmul": "x1, x2, /",
    "searchsorted": "x1, x2, /, *, side='left', sorter=None",
    "take_along_axis": "x, indices, /, *, axis=-1",
    "take": "x, indices, /, *, axis=None",
    "tensordot": "x1, x2, /, *, axes=2",
    "vecdot": "x1, x2, /, *, axis=-1",
}

THREE_ARG_FUNCS = {
    "where": "condition, x1, x2, /",
}

for func in UNARY_OPS:
    ONE_ARG_FUNCS[func] = "x, /"

for func in BINARY_OPS:
    TWO_ARG_FUNCS[func] = "x1, x2, /"

FFT_FUNCS = {
    'fft': 'x, /, *, n=None, axis=-1, norm="backward"',
    'ifft': 'x, /, *, n=None, axis=-1, norm="backward"',
    'fftn': 'x, /, *, s=None, axes=None, norm="backward"',
    'ifftn': 'x, /, *, s=None, axes=None, norm="backward"',
    'rfft': 'x, /, *, n=None, axis=-1, norm="backward"',
    'irfft': 'x, /, *, n=None, axis=-1, norm="backward"',
    'rfftn': 'x, /, *, s=None, axes=None, norm="backward"',
    'irfftn': 'x, /, *, s=None, axes=None, norm="backward"',
    'hfft': 'x, /, *, n=None, axis=-1, norm="backward"',
    'ihfft': 'x, /, *, n=None, axis=-1, norm="backward"',
    'fftshift': 'x, /, *, axes=None',
    'ifftshift': 'x, /, *, axes=None',
}

FFT_FREQ_FUNCS = {
    'fftfreq': 'n, /, *, d=1.0, dtype=None, device=None',
    'rfftfreq': 'n, /, *, d=1.0, dtype=None, device=None',
}

LINALG_ONE_ARG_FUNCS = {
    'cholesky': 'x, /, *, upper=False',
    'det': 'x, /',
    'diagonal': 'x, /, *, offset=0',
    'eigh': 'x, /',  # Tuple as output
    'eigvalsh': 'x, /',
    'inv': 'x, /',
    'matrix_norm': 'x, /, *, keepdims',
    'matrix_power': 'x, n, /',
    'matrix_rank': 'x, /, *, rtol=None',
    'matrix_transpose': 'x, /',
    'pinv': 'x, /, *, rtol=None',
    'qr': 'x, /, *, mode="reduced"',  # Tuple as output
    'slogdet': 'x, /',  # Tuple as output
    'svd': 'x, /, *, full_matrices=True',  # Tuple as output
    'svdvals': 'x, /',
    'trace': 'x, /, *, offset=0, dtype=None',
    'vector_norm': 'x, /, *, axis=None, keepdims=False, ord=2',
}

LINALG_TWO_ARG_FUNCS = {
    'cross': 'x1, x2, /, *, axis=-1',
    'matmul': 'x1, x2, /',
    'outer': 'x1, x2, /',
    'solve': 'x1, x2, /',
    'tensordot': 'x1, x2, /, *, axis=2',
    'vecdot': 'x1, x2, /, *, axis=-1',
}

def _make_namespace(slots):
    class Namespace:
        __slots__ = slots

    return Namespace()

# Global cache for FieldNamespace instances.
_field_namespace_cache = {}
_last_backend = None
_last_field_namespace = None

def make_field_namespace(backend):
    """
    Create a FieldNamespace object with all pre-bound Array API functions
    for a given backend.
    """
    global _field_namespace_cache
    global _last_backend
    global _last_field_namespace

    # Fast path in case of repeated calls to make_field_namespace().
    if backend is _last_backend:
        return _last_field_namespace

    # Look up in cache.
    key = id(backend)

    if key in _field_namespace_cache:
        namespace = _field_namespace_cache[key]

        _last_backend = backend
        _last_field_namespace = namespace

        return namespace

    # Create a new Field namespace.
    slots = ("__array_api_version__", "__name__")
    slots += tuple(CONSTANTS)
    slots += tuple(FEEDTHROUGH_FUNCS)
    slots += tuple(ZERO_ARG_FUNCS.keys())
    slots += tuple(ONE_ARG_FUNCS.keys())
    slots += tuple(TWO_ARG_FUNCS.keys())
    slots += tuple(THREE_ARG_FUNCS.keys())
    if hasattr(backend, 'fft'):
        slots += ('fft',)
    if hasattr(backend, 'linalg'):
        slots += ('linalg',)

    namespace = _make_namespace(slots)

    # Set Array API version.
    namespace.__array_api_version__ = backend.__array_api_version__
    namespace.__name__ = f'hcipy({backend.__name__})'

    # Set constants.
    for const_name in CONSTANTS:
        backend_const = getattr(backend, const_name)
        setattr(namespace, const_name, backend_const)

    # Set feedthrough functions.
    for func_name in FEEDTHROUGH_FUNCS:
        func = getattr(backend, func_name)
        setattr(namespace, func_name, func)

    # Set zero-array-argument functions.
    for func_name, sig in ZERO_ARG_FUNCS.items():
        func = getattr(backend, func_name)
        wrapper_func = _make_array_api_namespace_func(func, sig, num_array_args=0)
        setattr(namespace, func_name, wrapper_func)

    # Set one-array-argument functions.
    for func_name, sig in ONE_ARG_FUNCS.items():
        func = getattr(backend, func_name)
        wrapper_func = _make_array_api_namespace_func(func, sig, num_array_args=1)
        setattr(namespace, func_name, wrapper_func)

    # Set two-array-argument functions.
    for func_name, sig in TWO_ARG_FUNCS.items():
        func = getattr(backend, func_name)
        wrapper_func = _make_array_api_namespace_func(func, sig, num_array_args=2)
        setattr(namespace, func_name, wrapper_func)

    # Set three-array-argument functions.
    for func_name, sig in THREE_ARG_FUNCS.items():
        func = getattr(backend, func_name)
        wrapper_func = _make_array_api_namespace_func(func, sig, num_array_args=3)
        setattr(namespace, func_name, wrapper_func)

    # Add FFT extension if it exists on this backend.
    if hasattr(backend, 'fft'):
        slots = tuple(FFT_FUNCS.keys()) + tuple(FFT_FREQ_FUNCS.keys())
        fft_namespace = _make_namespace(slots)

        # Set the FFT functions.
        for func_name, sig in FFT_FUNCS.items():
            func = getattr(backend.fft, func_name)
            wrapper_func = _make_array_api_namespace_func(func, sig, num_array_args=1)
            setattr(fft_namespace, func_name, wrapper_func)

        # Set the FFT freq functions.
        for func_name, sig in FFT_FUNCS.items():
            func = getattr(backend.fft, func_name)
            wrapper_func = _make_array_api_namespace_func(func, sig, num_array_args=0)
            setattr(fft_namespace, func_name, wrapper_func)

        namespace.fft = fft_namespace

    # Add LinAlg extension if it exists on this backend.
    if hasattr(backend, 'linalg'):
        slots = tuple(LINALG_ONE_ARG_FUNCS.keys()) + tuple(LINALG_TWO_ARG_FUNCS.keys())
        linalg_namespace = _make_namespace(slots)

        # Set the one-arg functions.
        for func_name, sig in LINALG_ONE_ARG_FUNCS.items():
            func = getattr(backend.linalg, func_name)
            wrapper_func = _make_array_api_namespace_func(func, sig, num_array_args=1)
            setattr(linalg_namespace, func_name, wrapper_func)

        # Set the two-arg functions.
        for func_name, sig in LINALG_TWO_ARG_FUNCS.items():
            func = getattr(backend.linalg, func_name)
            wrapper_func = _make_array_api_namespace_func(func, sig, num_array_args=2)
            setattr(linalg_namespace, func_name, wrapper_func)

        namespace.linalg = linalg_namespace

    # Add new namespace to cache and return.
    _field_namespace_cache[key] = namespace

    _last_backend = backend
    _last_field_namespace = namespace

    return namespace

if Configuration().core.use_new_style_fields:
    Field = NewStyleField
else:
    Field = OldStyleField

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
    return isinstance(obj, FieldBase)
