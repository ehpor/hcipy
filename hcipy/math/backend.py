import importlib
import functools
from collections import defaultdict, OrderedDict

from ..config import Configuration

import numpy as onp

def first_arg_dispatcher(*args, **kwargs):
    return args[0]

_functions = defaultdict(lambda: {})
_dispatchers = defaultdict(lambda: first_arg_dispatcher)
_backend_aliases = {}

_module_aliases = {}
_function_aliases = {}
_function_wrappers = {}

_constants = [
    'e',
    'euler_gamma',
    'inf',
    'nan',
    'newaxis',
    'pi'
]

def call(func_name, *args, like=None, **kwargs):
    if like is None:
        backend = infer_backend_from_signature(func_name, *args, **kwargs)
    elif isinstance(like, str):
        backend = like
    else:
        backend = infer_backend_from_array(like)

    try:
        func = _functions[backend][func_name]
    except KeyError:
        func = import_library_function(backend, func_name)

    return func(*args, **kwargs)

class BackendShim:
    def __init__(self, submodule=None):
        self.submodule = submodule

        if submodule is None:
            self.linalg = BackendShim('linalg')
            self.random = BackendShim('random')
            self.fft = BackendShim('fft')

            for name in _constants:
                setattr(self, name, getattr(onp, name))

    def __getattr__(self, func_name):
        if self.submodule is not None:
            full_func_name = self.submodule + '.' + func_name
        else:
            full_func_name = func_name

        # Cache partial function on this object for performance.
        func_partial = functools.partial(call, full_func_name)
        setattr(self, func_name, func_partial)

        return func_partial

numpy = BackendShim()

def infer_backend_from_signature(func_name, *args, **kwargs):
    dispatch_argument = _dispatchers[func_name](*args, **kwargs)

    return infer_backend_from_array(dispatch_argument.__class__)

@functools.lru_cache(None)
def infer_backend_from_array(arr):
    module_name = arr.__module__.split('.')[0]

    if module_name in _backend_aliases:
        module_name = _backend_aliases[module_name]

    return _backend_aliases.get(module_name, module_name)

def import_library_function(backend, func_name):
    print('importing', func_name, 'from', backend)
    try:
        module_name = _module_aliases.get(backend, backend)

        full_func_name = _function_aliases.get((backend, func_name), func_name)
        full_func_name = module_name + '.' + full_func_name

        split_func_name = full_func_name.split('.')
        module_name = '.'.join(split_func_name[:-1])
        only_func_name = split_func_name[-1]

        try:
            library = importlib.import_module(module_name)
        except ImportError:
            if '.' in module_name:
                mod, submod = module_name.split('.')
                library = getattr(importlib.import_module(mod), submod)
            else:
                raise AttributeError

        library_function = getattr(library, only_func_name)

        if (module_name, only_func_name) in _function_wrappers:
            library_function = _function_wrappers[module_name, only_func_name](library_function)

        _functions[backend][func_name] = library_function

    except AttributeError:
        raise ImportError(f"Couldn't find function '{func_name}' for backend '{backend}'.")

    return library_function

def register_dispatcher(func_name):
    def decorator(func):
        if isinstance(func_name, str):
            _dispatchers[func_name] = func
        else:
            for name in func_name:
                _dispatchers[name] = func

        return func

    return decorator

def translate_wrapper(fn, translator):
    """Wrap a function to match the api of another according to a translation.
    The ``translator`` entries in the form of an ordered dict should have
    entries like:
        (desired_kwarg: (backend_kwarg, default_value))
    with the order defining the args of the function.
    """

    @functools.wraps(fn)
    def translated_function(*args, **kwargs):
        new_kwargs = {}
        translation = translator.copy()

        # convert args
        for arg_value in args:
            new_arg_name = translation.popitem(last=False)[1][0]
            new_kwargs[new_arg_name] = arg_value

        # convert kwargs -  but only those in the translation
        for key, value in kwargs.items():
            try:
                new_kwargs[translation.pop(key)[0]] = value
            except KeyError:
                new_kwargs[key] = value

        # set remaining default kwargs
        for key, value in translation.items():
            new_kwargs[value[0]] = value[1]

        return fn(**new_kwargs)

    return translated_function

def make_translator(t):
    return functools.partial(translate_wrapper, translator=OrderedDict(t))

# Dispatchers

@register_dispatcher(['concatenate', 'stack', 'block', 'vstack', 'hstack', 'dstack', 'column_stack', 'row_stack'])
def join_array_dispatcher(*args, **kwargs):
    try:
        return args[0][0]
    except (TypeError, ValueError):
        return args[0]

@register_dispatcher('einsum')
def einsum_dispatcher(*args, **kwargs):
    if isinstance(args[0], str):
        return args[1]
    else:
        return args[0]

_module_aliases['decimal'] = 'math'
_module_aliases['builtins'] = Configuration().core.default_backend
_module_aliases['hcipy'] = 'numpy'
