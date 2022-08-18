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
        # I'm sure there is a way to use deprecated() above, but I can't figure that out right now.
        @functools.wraps(new_func)
        def wrapped(*args, **kwargs):
            message = f'{old_func.__name__} is deprecated. Its new name is {new_func.__name__}.'
            warnings.warn(message, DeprecationWarning, stacklevel=2)

            return new_func(*args, **kwargs)

        return wrapped
    return decorator
