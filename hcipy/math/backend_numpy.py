from .backend import _functions, _custom_wrappers, call

def numpy_to_numpy(x):
    return call('asarray', x, like='numpy')

_functions['numpy']['to_numpy'] = numpy_to_numpy
_functions['builtins']['to_numpy'] = numpy_to_numpy
#_custom_wrappers['numpy', 'linalg.svd'] = svd_not_full_matrices_wrapper
#_custom_wrappers['numpy', 'random.normal'] = with_dtype_wrapper
#_custom_wrappers['numpy', 'random.uniform'] = with_dtype_wrapper
