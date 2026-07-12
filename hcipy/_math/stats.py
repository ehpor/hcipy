from array_api_compat import is_cupy_namespace, is_jax_namespace, is_numpy_namespace, is_torch_namespace
from .typing import Array, ArrayNamespace
from .backends import array_namespace
from typing import Any, cast

def median(x: Array, /, *, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Array:
    """Compute the median of an array along the given axis.

    Parameters
    ----------
    x : Array
        Input array (any supported library).
    axis : int | tuple[int, ...] | None
        Axis or axes to reduce over.  ``None`` flattens first.
    keepdims : bool
        If True, reduced axes are kept with size 1.

    Returns
    -------
    Array
        An array with the same type as `x`.
    """
    from ..field import NewStyleField
    if isinstance(x, NewStyleField):
        return NewStyleField(median(x.data, axis=axis, keepdims=keepdims), x.grid)

    xp = array_namespace(x)

    # NumPy, CuPy and JAX have a correct multi-axis median.
    if is_numpy_namespace(xp):
        import numpy as np
        res = np.median(cast(np.ndarray, x), axis=axis, keepdims=keepdims)
        return cast(Array, res)
    elif is_cupy_namespace(xp):
        import cupy as cp
        res = cp.median(cast(cp.ndarray, x), axis=axis, keepdims=keepdims)
        return cast(Array, res)
    elif is_jax_namespace(xp):
        import jax.numpy as jnp
        res = jnp.median(cast(jnp.ndarray, x), axis=axis, keepdims=keepdims)
        return cast(Array, res)
    elif is_torch_namespace(xp):
        return _median_torch(x, axis=axis, keepdims=keepdims)
    else:
        raise NotImplementedError("Unsupported backend.")

def nanmedian(x: Array, /, *, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Array:
    """Compute the median of an array along the given axis.

    Parameters
    ----------
    x : Array
        Input array (any supported library).
    axis : int | tuple[int, ...] | None
        Axis or axes to reduce over.  ``None`` flattens first.
    keepdims : bool
        If True, reduced axes are kept with size 1.

    Returns
    -------
    Array
        An array with the same type as `x`.
    """
    from ..field import NewStyleField
    if isinstance(x, NewStyleField):
        return NewStyleField(nanmedian(x.data, axis=axis, keepdims=keepdims), x.grid)

    xp = array_namespace(x)

    # NumPy, CuPy and JAX have a correct multi-axis median.
    if is_numpy_namespace(xp):
        import numpy as np
        res = np.nanmedian(cast(np.ndarray, x), axis=axis, keepdims=keepdims)
        return cast(Array, res)
    elif is_cupy_namespace(xp):
        import cupy as cp
        res = cp.nanmedian(cast(cp.ndarray, x), axis=axis, keepdims=keepdims)
        return cast(Array, res)
    elif is_jax_namespace(xp):
        import jax.numpy as jnp
        res = jnp.nanmedian(cast(jnp.ndarray, x), axis=axis, keepdims=keepdims)
        return cast(Array, res)
    elif is_torch_namespace(xp):
        return _nanmedian_torch(x, axis=axis, keepdims=keepdims)
    else:
        raise NotImplementedError("Unsupported backend.")

def _reshape_last_axis(x: Array, *, xp: ArrayNamespace[Any], axis: int | tuple[int, ...] | None) -> tuple[Array, Any, Any]:
    ndim = x.ndim

    if ndim == 0:
        return x, None, None

    if axis is None:
        axes: tuple[int, ...] | None = None
    elif isinstance(axis, int):
        axes = (axis % ndim,)
    else:
        axes = tuple(a % ndim for a in axis)

    if axes is not None and len(axes) == 0:
        return x, None, None

    # Single axis: partition along that axis. Multi-axis/None: flatten to last axis.
    if axes is None or len(axes) > 1:
        partition_axis = -1
    else:
        partition_axis = axes[0]

    # Flatten reduced axes into a single axis.
    if axes is None:
        x = xp.reshape(x, (-1,))
    elif len(axes) > 1:
        other_axes = tuple(i for i in range(ndim) if i not in axes)
        perm = other_axes + tuple(sorted(axes))
        x = xp.permute_dims(x, perm)

        keep_shape = x.shape[:len(other_axes)]
        x = xp.reshape(x, keep_shape + (-1,))

    return x, axes, partition_axis

def _reshape_keepdims(x: Array, axes: tuple[int, ...] | None, keepdims: bool, ndim: int, xp: ArrayNamespace[Any]) -> Array:
    if keepdims:
        if axes is not None:
            x = xp.expand_dims(x, axis=axes)
        else:
            x = xp.reshape(x, (1,) * ndim)

    return x

def _median_torch(x: Array, axis: int | tuple[int, ...] | None, keepdims: bool) -> Array:
    import torch
    x_reshaped, axes, partition_axis = _reshape_last_axis(x, xp=torch, axis=axis)

    if partition_axis is None:
        return x_reshaped

    if torch.isdtype(x_reshaped.dtype, "integral"):
        x_reshaped = torch.astype(x_reshaped, torch.float64)

    res = torch.quantile(x_reshaped, 0.5, dim=partition_axis)

    return _reshape_keepdims(res, axes, keepdims, x.ndim, torch)

def _nanmedian_torch(x: Array, axis: int | tuple[int, ...] | None, keepdims: bool) -> Array:
    import torch
    x_reshaped, axes, partition_axis = _reshape_last_axis(x, xp=torch, axis=axis)

    if partition_axis is None:
        return x_reshaped

    if torch.isdtype(x_reshaped.dtype, "integral"):
        x_reshaped = torch.astype(x_reshaped, torch.float64)

    res = torch.nanquantile(x_reshaped, 0.5, dim=partition_axis)

    return _reshape_keepdims(res, axes, keepdims, x.ndim, torch)
