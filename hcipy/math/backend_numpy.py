import numpy as onp
from .backend import _functions, call
from .primitives import primitive_function

def numpy_to_numpy(x):
    return call('asarray', x, like='numpy')

def numpy_loadtxt(*args, **kwargs):
    from numpy import loadtxt
    return call('assarray', loadtxt(*args, **kwargs))

_functions['numpy']['to_numpy'] = numpy_to_numpy
_functions['builtins']['to_numpy'] = numpy_to_numpy
_functions['_io']['loadtxt'] = onp.loadtxt

@primitive_function('numpy')
def jones_matrix_to_q(jones_matrix):
    return 'numpy'
