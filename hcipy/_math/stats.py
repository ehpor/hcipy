from array_api_compat import is_cupy_namespace, is_jax_namespace, is_numpy_namespace, is_torch_namespace
from .backends import array_namespace

def median(x, /, *, axis=None, keepdims=False):
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
    if is_numpy_namespace(xp) or is_cupy_namespace(xp) or is_jax_namespace(xp):
        return xp.median(x, axis=axis, keepdims=keepdims)

    # Torch needs a specific implementation.
    if is_torch_namespace(xp):
        return _median_torch(x, axis=axis, keepdims=keepdims, xp=xp)

    raise NotImplementedError("Unsupported backend.")

def nanmedian(x, /, *, axis=None, keepdims=False):
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
    if is_numpy_namespace(xp) or is_cupy_namespace(xp) or is_jax_namespace(xp):
        return xp.nanmedian(x, axis=axis, keepdims=keepdims)

    # Torch needs a specific implementation.
    if is_torch_namespace(xp):
        return _nanmedian_torch(x, axis=axis, keepdims=keepdims, xp=xp)

    raise NotImplementedError("Unsupported backend.")

def _reshape_last_axis(x, *, xp, axis):
    ndim = x.ndim

    if ndim == 0:
        return x, None, None

    if axis is None:
        axes = None
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

def _reshape_keepdims(x, axes, keepdims, ndim, xp):
    if keepdims:
        if axes is not None:
            x = xp.expand_dims(x, axis=axes)
        else:
            x = xp.reshape(x, (1,) * ndim)

    return x

def _median_torch(x, axis, keepdims, xp):
    x_reshaped, axes, partition_axis = _reshape_last_axis(x, xp=xp, axis=axis)

    if partition_axis is None:
        return x_reshaped

    if xp.isdtype(x_reshaped.dtype, "integral"):
        x_reshaped = xp.astype(x_reshaped, xp.float64)

    res = xp.quantile(x_reshaped, 0.5, dim=partition_axis)

    return _reshape_keepdims(res, axes, keepdims, x.ndim, xp)

def _nanmedian_torch(x, axis, keepdims, xp):
    x_reshaped, axes, partition_axis = _reshape_last_axis(x, xp=xp, axis=axis)

    if partition_axis is None:
        return x_reshaped

    if xp.isdtype(x_reshaped.dtype, "integral"):
        x_reshaped = xp.astype(x_reshaped, xp.float64)

    res = xp.nanquantile(x_reshaped, 0.5, dim=partition_axis)

    return _reshape_keepdims(res, axes, keepdims, x.ndim, xp)
