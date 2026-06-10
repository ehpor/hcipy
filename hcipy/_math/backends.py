import numpy as np
import array_api_compat
from ..config import Configuration
import sys

def infer_xp(*arrays):
    """Infer xp from input arrays.

    Parameters
    ----------
    *arrays : arrays
        Variable number of input arrays to infer xp from

    Returns
    -------
    xp : module
        The inferred xp module, or numpy as fallback (in legacy mode)

    Raises
    ------
    ValueError
        If xp cannot be inferred and use_new_style_fields is True
    """
    # Filter out None values
    arrays = tuple(a for a in arrays if a is not None)

    if len(arrays) > 0:
        try:
            return array_api_compat.array_namespace(*arrays)
        except (ValueError, TypeError):
            pass

    # If we get here, xp could not be inferred
    if Configuration().core.use_new_style_fields:
        raise ValueError("xp must be specified when arrays don't provide it")
    return np

def to_numpy(arr):
    """Convert any array to numpy.

    This function uses backend-specific methods for efficient conversion. It
    handles device synchronization for GPU arrays.

    Parameters
    ----------
    arr : array
        Input array from any backend (numpy, cupy, jax, etc.)

    Returns
    -------
    np.ndarray
        Numpy array containing the same data

    Raises
    ------
    ValueError
        If conversion fails or array type is not supported
    """
    # Fast path: already numpy
    if array_api_compat.is_numpy_array(arr):
        return np.asarray(arr)

    # CuPy: use .get() method
    if array_api_compat.is_cupy_array(arr):
        return arr.get()

    # JAX: np.asarray works but transfers from device
    if array_api_compat.is_jax_array(arr):
        return np.asarray(arr)

    # PyTorch: use .numpy() method (requires detach and cpu)
    if array_api_compat.is_torch_array(arr):
        return arr.detach().cpu().numpy()

    # Dask: compute first
    if array_api_compat.is_dask_array(arr):
        return arr.compute()

    # Fallback: try np.asarray
    try:
        return np.asarray(arr)
    except Exception as e:
        raise ValueError(f"Cannot convert {type(arr).__name__} to numpy: {e}")

def is_scalar(element):
    '''Check whether an element is a scalar.

    Parameters
    ----------
    element : any
        The element to check.

    Returns
    -------
    bool
        Whether the element is a scalar.
    '''
    ndim = getattr(element, 'ndim', None)
    if ndim is None:
        # Plain Python scalar, no array structure
        return True

    shape = getattr(element, 'shape', ())
    return all(n == 1 for n in shape)

def all_close(a, b, *, rtol=1e-5, atol=1e-8):
    '''Whether the two arrays are elementwise equal within tolerance.

    This is defined as `|a - b| <= atol + rtol * |b|`.

    At least `a` or `b` needs to be an Array.

    Parameters
    ----------
    a : Array or scalar
        The first array or scalar.
    b : Array or scalar
        The second array or scalar.
    rtol : float, optional
        _description_, by default 1e-5
    atol : float, optional
        _description_, by default 1e-8
    '''
    xp = array_namespace(a, b)

    close = abs(a - b) <= atol + rtol * abs(b)
    return xp.all(close)

namespace_caches = {}
NUMPY_NAMESPACE = 'numpy'

def array_namespace(*xs, api_version=None):
    '''Return the namespace of a set of Arrays.

    Parameters
    ----------
    *xs : tuple of Arrays
        The array objects of which to find the namespace.
    api_version : str, optional
        The Array API spec version. If this is None (default),
        the choice is left to the Array object.

    Returns
    -------
    module
        The Array namespace.
    '''
    from ..field import NewStyleField, make_field_namespace

    try:
        cache = namespace_caches[api_version]
    except KeyError:
        namespace_caches[api_version] = {}
        cache = namespace_caches[api_version]

    for x in xs:
        if isinstance(x, NewStyleField):
            return make_field_namespace(array_namespace(x.data, api_version=api_version))

        try:
            ns = cache[type(x)]

            if ns == NUMPY_NAMESPACE:
                # Could be either a Numpy array or
                # a JAX zero gradient array.
                if 'jax' in sys.modules:
                    import jax

                    if x.dtype == jax.float0:
                        import jax.numpy as jnp
                        return jnp
                return np

            return ns
        except KeyError:
            continue

    for x in xs:
        try:
            ns = array_api_compat.array_namespace(x, api_version=api_version)

            # Numpy is a special case, see above.
            if x.__class__.__module__.startswith('numpy'):
                cache[type(x)] = NUMPY_NAMESPACE
            else:
                cache[type(x)] = ns

            return array_namespace(x, api_version=api_version)
        except KeyError:
            cache[type(x)] = None

    return array_api_compat.array_namespace(*xs, api_version=api_version)

def default_dtype(xp, kind, device=None):
    '''Find the default dtype of a certain kind.

    Parameters
    ----------
    xp : module
        The Array API backend.
    kind : {"real floating", "complex floating", "integral", "indexing"}
        The kind of dtype.
    device : Device or None
        The device on the Array API backend. If not given, the default device
        of the backend will be used.

    Returns
    -------
    dtype
        The default dtype.
    '''
    return xp.__array_namespace_info__().default_dtypes(device=device)[kind]
