from .backend import infer_backend_from_array
from collections import defaultdict
from functools import wraps
from inspect import signature

_primitive_functions = defaultdict(lambda: {})

def primitive_function(backend=None):
    def decorator(func):
        # Add the function to the primitive functions database.
        _primitive_functions[backend][func.__name__] = func

        # Check that the signature matches that of the fallback function.
        fallback_func = get_primitive_function(func.__name__)
        if signature(fallback_func) != signature(func):
            raise ValueError(f"Function '{func.__name__}' on backend {backend} has a different signature than the fallback function.")

        if backend is not None:
            return func

        @wraps(func)
        def wrapper(*args, **kwargs):
            backend = infer_backend_from_array(args[0].__class__)

            f = get_primitive_function(func.__name__, backend)
            return f(*args, **kwargs)

        return wrapper

    return decorator

def get_primitive_function(func_name, backend=None):
    if func_name in _primitive_functions[backend]:
        return _primitive_functions[backend][func_name]
    else:
        return _primitive_functions[None][func_name]
