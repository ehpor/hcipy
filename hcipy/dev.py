import functools
import warnings

def deprecated(reason):  # pragma: no cover
    '''Decorator for deprecating functions.

    Parameters
    ----------
    reason : str
        The reason for deprecation. This will be printed with the warning.

    Returns
    -------
    function
        The new function that emits a DeprecationWarning upon use.
    '''
    def decorator(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            warnings.warn(f'{func.__name__} is deprecated. {reason}', DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapped
    return decorator

def deprecated_name_changed(new_func):  # pragma: no cover
    '''Decorator for deprecating functions with a name change.

    Parameters
    ----------
    new_func : function
        The new function with the same functionality.

    Returns
    -------
    function
        The new function that emits a DeprecationWarning upon use.
    '''
    def decorator(old_func):
        f = new_func
        f.__name__ = old_func.__name__

        return deprecated(f'Its new name is {new_func.__name__}.')(f)
    return decorator
