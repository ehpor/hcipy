from .backend import _functions, _module_aliases, _backend_aliases, call
from .primitives import primitive_function

_jax_random_key = None

def jax_default_rng(seed=None):
    #TODO a jax equivalent to numpy.random.default_rng
    raise NotImplementedError

def jax_random_seed(seed=None):
    from jax.random import PRNGKey

    global _jax_random_key

    if seed is None:
        from random import SystemRandom

        seed = SystemRandom().randint(-(2 ** 63), 2 ** 63 - 1)

    _jax_random_key = PRNGKey(seed)

def jax_random_get_key():
    from jax.random import split

    global _jax_random_key

    if _jax_random_key is None:
        jax_random_seed()

    _jax_random_key, subkey = split(_jax_random_key)

    return subkey

def jax_random_uniform(low=0.0, high=1.0, size=None, **kwargs):
    from jax.random import uniform

    if size is None:
        size = ()

    return uniform(jax_random_get_key(), shape=size, minval=low, maxval=high, **kwargs)

def jax_random_normal(loc=0.0, scale=1.0, size=None, **kwargs):
    from jax.random import normal

    if size is None:
        size = ()

    x = normal(jax_random_get_key(), shape=size, **kwargs)

    if scale != 1.0:
        x *= scale

    if loc != 0.0:
        x += loc

    return x

def jax_random_poisson(lam=1.0, size=None, **kwargs):
    from jax.random import poisson

    if size is None:
        size = ()

    x = poisson(jax_random_get_key(), lam, shape=size, **kwargs)

    return x

def jax_random_gamma(shape, scale=1.0, size=None, **kwargs):
    #TODO JAX doesn't use this scale parameter. We should implement different scaling ourselves
    from jax.random import gamma
    
    if size is None:
        size = ()

    x = gamma(jax_random_get_key(), shape, shape=size, **kwargs)
    
    return x

def jax_random_exponential(scale=1.0, size=None, **kwargs):
    from jax.random import exponential

    if size is None:
        size = ()

    x = exponential(jax_random_get_key(), shape=size, **kwargs)

    if scale != 1.0:
        x *= scale

    return x

def jax_random_randint(low, high=None, size=None, **kwargs):
    from jax.random import randint

    if size is None:
        size = ()

    if high is None:
        high = low
        low = 0

    return randint(jax_random_get_key(), shape=size, minval=low, maxval=high, **kwargs)

def jax_random_choice(a, size=None, replace=True, **kwargs):
    from jax.random import choice

    if size is None:
        size = ()

    return choice(jax_random_get_key(), a, shape=size, replace=replace, **kwargs)

def jax_random_rand(*args, **kwargs):
    return jax_random_uniform(**kwargs, size=[di for di in args])

def jax_random_randn(*args, **kwargs):
    return jax_random_normal(**kwargs, size=[di for di in args])

def jax_to_numpy(x):
    return call('asarray', x, like='numpy')

_backend_aliases['jaxlib'] = 'jax'

_module_aliases['jax'] = 'jax.numpy'

_functions['jax']['to_numpy'] = jax_to_numpy

# random libary needs to be defined for both builtins and jax
_functions['builtins']['random.default_rng'] = jax_default_rng
_functions['builtins']['random.seed'] = jax_random_seed
_functions['builtins']['random.uniform'] = jax_random_uniform
_functions['builtins']['random.normal'] = jax_random_normal
_functions['builtins']['random.rand'] = jax_random_rand
_functions['builtins']['random.randn'] = jax_random_randn
_functions['builtins']['random.randint'] = jax_random_randint
_functions['builtins']['random.choice'] = jax_random_choice
_functions['builtins']['random.gamma'] = jax_random_gamma
_functions['builtins']['random.exponential'] = jax_random_exponential
_functions['builtins']['random.poisson'] = jax_random_poisson

_functions['jax']['random.uniform'] = jax_random_uniform
_functions['jax']['random.normal'] = jax_random_normal
_functions['jax']['random.rand'] = jax_random_rand
_functions['jax']['random.randn'] = jax_random_randn
_functions['jax']['random.randint'] = jax_random_randint
_functions['jax']['random.choice'] = jax_random_choice
_functions['jax']['random.gamma'] = jax_random_gamma
_functions['jax']['random.exponential'] = jax_random_exponential
_functions['jax']['random.poisson'] = jax_random_poisson


@primitive_function('jax')
def jones_matrix_to_q(jones_matrix):
    return 'jax'
