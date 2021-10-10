import numpy as np

from ..config import Configuration

class Field(object):
    backend = None

    _backends = {}
    _backend_aliases = {}

    @classmethod
    def resolve_backend(cls, backend):
        if backend in cls._backend_aliases:
            return cls._backend_aliases[backend]

        if backend not in cls._backends:
            raise KeyError(f'Backend "{backend}" not implemented.')

        return backend

    def __new__(cls, array, grid, backend=None):
        # Avoid infinite recursion due to children inheriting this function as well.
        if cls is not Field:
            return super().__new__(cls)

        if backend is None:
            # Use default backend unless overridden.
            backend = Configuration().field.default_backend

            if is_field(array):
                # If array is already a Field, just create a new Field with the same backend.
                backend = array.backend
            else:
                # Decide backend on array_type.
                for name, backend_class in Field._backends.items():
                    if backend_class.is_native_array(array):
                        backend = name
                        break
        else:
            if is_field(array):
                if array.backend != backend:
                    array = array.numpy().array

        # Return Field of the correct backend
        backend = Field.resolve_backend(backend)

        return Field._backends[backend](array, grid)

    @classmethod
    def is_native_array(cls, array):
        raise NotImplementedError()

    def as_backend(self, backend):
        backend = self.resolve_backend(backend)

        if self.backend == backend:
            return self

        # Convert
        return self._backends[backend](self.numpy().array, self.grid)

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

    def __repr__(self):
        '''The text representation of the field.

        Returns
        -------
        str
            The text representation of the field.
        '''
        return self.__class__.__name__ + '(' + repr(self.array) + ')'

    def __str__(self):
        '''The field converted to a string.

        Returns
        -------
        str
            The field converted to a string.
        '''
        return self.__class__.__name__ + '(' + str(self.array) + ')'

def field_backend(name, aliases=None):
    if aliases is None:
        aliases = []

    def decorator(cls):
        if not issubclass(cls, Field):
            raise ValueError('All Fields must be a subclass of Field.')

        cls.backend = name

        if name in Field._backends:
            raise ValueError('Backend already registered.')

        Field._backends[name] = cls

        for alias in aliases:
            if alias in Field._backends:
                raise ValueError('A backend already exists with this name.')

            if alias in Field._backend_aliases:
                raise ValueError('A backend already exists with this alias.')

            Field._backend_aliases[alias] = name

        setattr(Field, name, lambda self: self.as_backend(name))

        return cls

    return decorator

def is_field(obj):
    return isinstance(obj, Field)
