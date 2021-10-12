from .backend import _functions, _function_wrappers, call

def numpy_to_numpy(x):
    return call('asarray', x, like='numpy')

_functions['numpy']['to_numpy'] = numpy_to_numpy
_functions['builtins']['to_numpy'] = numpy_to_numpy
#_function_wrappers['numpy', 'linalg.svd'] = svd_not_full_matrices_wrapper
#_function_wrappers['numpy', 'random.normal'] = with_dtype_wrapper
#_function_wrappers['numpy', 'random.uniform'] = with_dtype_wrapper
