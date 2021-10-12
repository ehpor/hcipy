from .backend import _functions, _custom_wrappers

def cupy_to_numpy(x):  # pragma: no cover
    return x.get()

_functions['cupy']['to_numpy'] = cupy_to_numpy
#_custom_wrappers['cupy', 'linalg.svd'] = svd_not_full_matrices_wrapper
