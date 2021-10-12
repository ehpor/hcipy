from .backend import _functions, _custom_wrappers, _module_aliases, _backend_aliases, call

_jax_random_key = None

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

def jax_to_numpy(x):
    return call('asarray', x, like='numpy')

_backend_aliases['jaxlib'] = 'jax'

_module_aliases['jax'] = 'jax.numpy'

#_custom_wrappers['jax', 'linalg.qr'] = qr_allow_fat
#_custom_wrappers['jax', 'linalg.svd'] = svd_not_full_matrices_wrapper

_functions['jax']['to_numpy'] = jax_to_numpy
_functions['jax']['random.seed'] = jax_random_seed
_functions['jax']['random.uniform'] = jax_random_uniform
_functions['jax']['random.normal'] = jax_random_normal
