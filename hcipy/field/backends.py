import numpy as np
import array_api_compat
from ..config import Configuration


def _infer_xp(*arrays):
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
