from .backend import _functions, call
from .primitives import primitive_function

def numpy_to_numpy(x):
    return call('asarray', x, like='numpy')

_functions['numpy']['to_numpy'] = numpy_to_numpy
_functions['builtins']['to_numpy'] = numpy_to_numpy

@primitive_function('numpy')
def jones_matrix_to_q(jones_matrix):
    return 'numpy'
