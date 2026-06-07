from array_api_compat import is_cupy_namespace, is_jax_namespace, is_numpy_namespace
import array_api_extra as xpx
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

    # Fallback for Pytorch and unknown backends.
    return _median_via_partition(x, xp=xp, axis=axis, keepdims=keepdims)

def _median_via_partition(x, *, xp, axis, keepdims):
    """Median using only Array API functions.
    """
    ndim = x.ndim

    if ndim == 0:
        return x

    if axis is None:
        axes = None
        partition_axis = -1
    elif isinstance(axis, int):
        axes = (axis % ndim,)
        partition_axis = axis % ndim
    else:
        axes = tuple(a % ndim for a in axis)

    if axes is not None and len(axes) == 0:
        return x

    # Flatten reduced axes into a single axis.
    if axes is None:
        x = xp.reshape(x, (-1,))
        partition_axis = -1
    elif len(axes) > 1:
        other_axes = tuple(i for i in range(ndim) if i not in axes)
        perm = other_axes + tuple(sorted(axes))
        x = xp.permute_dims(x, perm)

        keep_shape = tuple(x.shape[i] for i in range(len(other_axes)))
        x = xp.reshape(x, keep_shape + (-1,))
        partition_axis = -1

    n = x.shape[partition_axis]
    mid = n // 2

    if n % 2 == 1:
        # Odd N: single partition at the middle.
        partitioned = xpx.partition(x, mid, axis=partition_axis, xp=xp)
        # Index along the partition axis
        idx = [slice(None)] * partitioned.ndim
        idx[partition_axis] = mid
        result = partitioned[tuple(idx)]
    else:
        # Even N: partition at mid-1 and mid. array_api_extra.partition only
        # accepts a single kth, so we call it twice.
        partitioned_lo = xpx.partition(x, mid - 1, axis=partition_axis, xp=xp)
        idx = [slice(None)] * partitioned_lo.ndim
        idx[partition_axis] = mid - 1
        lo = partitioned_lo[tuple(idx)]

        partitioned_hi = xpx.partition(x, mid, axis=partition_axis, xp=xp)
        idx[partition_axis] = mid
        hi = partitioned_hi[tuple(idx)]

        result = (lo + hi) / 2

    if keepdims:
        if axes is not None:
            result = xp.expand_dims(result, axis=axes)
        else:
            result = xp.reshape(result, (1,) * ndim)

    return result
