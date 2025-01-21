from .backend import _functions

def cupy_to_numpy(x):  # pragma: no cover
    return x.get()

_functions['cupy']['to_numpy'] = cupy_to_numpy
