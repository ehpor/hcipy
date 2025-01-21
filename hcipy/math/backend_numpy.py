from .backend import _functions, call

def numpy_to_numpy(x):
    return call('asarray', x, like='numpy')

_functions['numpy']['to_numpy'] = numpy_to_numpy
_functions['builtins']['to_numpy'] = numpy_to_numpy

